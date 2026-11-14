"""Flush subnormal numbers to zero inside the layer kernels.

Where the wave has not arrived the displacement is tiny but not zero: the explicit layer carries a number one
element further at every stage, and ahead of the wave it decays faster than exponentially. Below about 1e-154 the
products of strains fall under the smallest normal double (2.2e-308), and below 2.2e-308 the displacement itself
does. Arithmetic on such subnormal numbers is handled by the processor's microcode on x86 and takes many times
longer than on normal numbers (docs/large-models.md 7.5): the first microseconds of a record on a long plate ran
at a third of the normal speed, and the cores scaled worse.

With the processor's flush-to-zero and denormals-are-zero modes a result below 2.2e-308 becomes an exact zero and a
subnormal operand counts as zero. Nothing else changes: a number that is added to a normal number of ordinary
size was lost in rounding already, so only values that are themselves next to the bottom of the double range
can differ from the unflushed computation (measured on a 200 mm plate: displacements below 1e-285 m during the
first microsecond, by less than 1e-300 m, against signals of 1e-9 m; identical bits afterwards).

The mode belongs to a thread (the MXCSR register), so every task of a parallel region sets it on entry and puts
the previous value back when it leaves: `saved = enter(flush)` … `leave(saved)`. Other code of the thread is not
affected. `flush` false does nothing, which is the default everywhere: the frozen behaviour, bit for bit.

Only x86 is switched (`AVAILABLE`). On other processors `enter` does nothing and the record says so
(`flush_subnormals` false in the manifest).
"""
import platform
from numba import njit, types
from numba.extending import intrinsic

AVAILABLE = platform.machine().lower() in ('x86_64', 'amd64')
FLUSH_BITS = 0x8040   # MXCSR: flush-to-zero (bit 15) and denormals-are-zero (bit 6)

if AVAILABLE:
    from llvmlite import ir

    def _register(builder, name):
        return builder.module.declare_intrinsic(name, fnty=ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()]))

    @intrinsic
    def _read(typingctx):
        """The MXCSR register of this thread."""
        def codegen(context, builder, signature, args):
            slot = builder.alloca(ir.IntType(32))
            builder.call(_register(builder, 'llvm.x86.sse.stmxcsr'), [builder.bitcast(slot, ir.IntType(8).as_pointer())])
            return builder.load(slot)
        return types.uint32(), codegen

    @intrinsic
    def _write(typingctx, value):
        """Set the MXCSR register of this thread."""
        def codegen(context, builder, signature, args):
            slot = builder.alloca(ir.IntType(32)); builder.store(args[0], slot)
            builder.call(_register(builder, 'llvm.x86.sse.ldmxcsr'), [builder.bitcast(slot, ir.IntType(8).as_pointer())])
            return context.get_dummy_value()
        return types.void(types.uint32), codegen

    @njit(cache=True)
    def enter(flush):
        """Switch this thread to flushing when `flush`; returns what `leave` needs to put the previous mode back."""
        if not flush: return -1
        saved = _read(); _write(saved | types.uint32(FLUSH_BITS))
        return types.int64(saved)

    @njit(cache=True)
    def leave(saved):
        """Put back the mode `enter` found."""
        if saved >= 0: _write(types.uint32(saved))

    @njit(cache=True)
    def flushing():
        """Whether this thread flushes subnormal numbers now."""
        return (_read() & types.uint32(FLUSH_BITS)) == types.uint32(FLUSH_BITS)
else:
    @njit(cache=True)
    def enter(flush):
        return -1

    @njit(cache=True)
    def leave(saved):
        pass

    @njit(cache=True)
    def flushing():
        return False
