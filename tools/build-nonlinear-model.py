"""Reproducible additive revision of the recovered COMSOL Java source."""
from pathlib import Path
import hashlib
import sys

src = Path(sys.argv[1]).read_text()
assert hashlib.sha256(src.encode()).hexdigest() == 'ac090340bd65c47240b7954d2a51125b010f18d0e608bed012a6525febacccd7'
extra = r'''
    // WORKAI_MURNAGHAN_TOTAL_LAGRANGIAN_V1. Reference coordinates and dead
    // nominal traction; plane strain in z, with an optional anti-plane w field.
    private static boolean nonlinear=false;
    private static final String NL_SCHEMA="workai.murnaghan.guided_plate.scene.v1";
    private static final String[] CONTROLS={"linear_A","geometric_A","material_A","geometric_2A","material_2A"};
    private static String sha(String s)throws Exception{
        byte[] bytes=java.security.MessageDigest.getInstance("SHA-256").digest(s.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        StringBuilder out=new StringBuilder();for(byte b:bytes)out.append(String.format(Locale.ROOT,"%02x",b&255));return out.toString();
    }
    // Bounded JSON reader for the signed request: duplicate keys, nonfinite
    // numbers, invalid escapes/trailing content and excess nesting are rejected.
    private static class Json {
        String s;int at=0;Json(String s){this.s=s;if(s.length()>65536)bad();}
        void bad(){throw new IllegalArgumentException("Invalid request JSON");}
        void ws(){while(at<s.length()&&" \r\n\t".indexOf(s.charAt(at))>=0)at++;}
        char take(){if(at>=s.length())bad();return s.charAt(at++);}
        String str(){if(take()!='"')bad();StringBuilder b=new StringBuilder();while(true){char c=take();if(c=='"')return b.toString();if(c<32)bad();if(c=='\\'){c=take();switch(c){case '"':case '\\':case '/':b.append(c);break;case 'b':b.append('\b');break;case 'f':b.append('\f');break;case 'n':b.append('\n');break;case 'r':b.append('\r');break;case 't':b.append('\t');break;case 'u':if(at+4>s.length())bad();try{b.append((char)Integer.parseInt(s.substring(at,at+4),16));}catch(Exception e){bad();}at+=4;break;default:bad();}}else b.append(c);}}
        Object val(int depth){if(depth>12)bad();ws();if(at>=s.length())bad();char c=s.charAt(at);
            if(c=='"')return str();
            if(c=='{'){at++;Map<String,Object> m=new LinkedHashMap<>();ws();if(at<s.length()&&s.charAt(at)=='}'){at++;return m;}while(true){ws();String k=str();ws();if(take()!=':'||m.containsKey(k))bad();m.put(k,val(depth+1));ws();char end=take();if(end=='}')return m;if(end!=',')bad();}}
            if(c=='['){at++;List<Object> a=new ArrayList<>();ws();if(at<s.length()&&s.charAt(at)==']'){at++;return a;}while(true){a.add(val(depth+1));ws();char end=take();if(end==']')return a;if(end!=',')bad();}}
            if(s.startsWith("true",at)){at+=4;return Boolean.TRUE;}if(s.startsWith("false",at)){at+=5;return Boolean.FALSE;}if(s.startsWith("null",at)){at+=4;return null;}
            int start=at;while(at<s.length()&&"-+0123456789.eE".indexOf(s.charAt(at))>=0)at++;String tok=s.substring(start,at);
            if(!tok.matches("-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?"))bad();double v=Double.parseDouble(tok);if(!Double.isFinite(v))bad();return v;
        }
        Object read(){Object v=val(0);ws();if(at!=s.length())bad();return v;}
    }
    @SuppressWarnings("unchecked") private static Map<String,Object> obj(Object v){if(!(v instanceof Map))throw new IllegalArgumentException("JSON object required");return (Map<String,Object>)v;}
    private static void keys(Map<String,Object> v,String... k){if(!v.keySet().equals(new HashSet<String>(Arrays.asList(k))))throw new IllegalArgumentException("Scene fields");}
    private static void eq(Object a,Object b){if(!Objects.equals(a,b))throw new IllegalArgumentException("Scene identity/value mismatch");}
    private static double num(Map<String,Object> v,String k){Object x=v.get(k);if(!(x instanceof Double))throw new IllegalArgumentException("Numeric scene value required");return (Double)x;}
    private static void put(Map<String,Object> v,String k,Object x){if(x==null)throw new IllegalArgumentException("Missing scene field");v.put(k,x);}
    private static void readScene()throws Exception{
        Map<String,Object> request=obj(new Json(new String(Files.readAllBytes(Paths.get("request.json")),java.nio.charset.StandardCharsets.UTF_8)).read());
        for(String k:new String[]{"job_id","source_sha256","scene_sha256","parameters_sha256"})eq(request.get(k),p.getProperty(k));
        String sj=(String)request.get("scene_json"),pj=(String)request.get("parameters_json");eq(sha(sj),p.getProperty("scene_sha256"));eq(sha(pj),p.getProperty("parameters_sha256"));
        Map<String,Object> s=obj(new Json(sj).read());if(!NL_SCHEMA.equals(s.get("schema")))return;
        nonlinear=true;
        keys(s,"schema","constitutive","geometry","material","material_reference","boundary","source","receivers","acquisition","numerics","nonlinear");
        eq(s.get("constitutive"),"murnaghan_green_lagrange");eq(s.get("boundary"),"free");
        Map<String,Object> ref=obj(s.get("material_reference"));keys(ref,"id","record_sha256","usage");
        eq(ref.get("id"),"aluminum-polycrystal-blaschke2017-v1");eq(ref.get("record_sha256"),"39102235cd2fae341a277e46aa08573ae2294710c79abd104b10491fe5edef0a");eq(ref.get("usage"),"murnaghan_total_lagrangian");
        Map<String,Object> law=obj(s.get("nonlinear"));keys(law,"definition","lambda_pa","mu_pa","l_pa","m_pa","n_pa","controls");
        eq(law.get("definition"),"murnaghan_green_lagrange_reference_volume");eq(law.get("controls"),Arrays.asList(CONTROLS));
        String[] ks={"lambda_pa","mu_pa","l_pa","m_pa","n_pa"};double[] vs={58.1e9,26.1e9,-143e9,-297e9,-345e9};for(int j=0;j<ks.length;j++)if(num(law,ks[j])!=vs[j])throw new IllegalArgumentException("Unreviewed material constants");
        Map<String,Object> g=obj(s.get("geometry")),mat=obj(s.get("material")),acq=obj(s.get("acquisition")),disc=obj(s.get("numerics")),tx=obj(s.get("source"));
        keys(g,"length_mm","thickness_mm");keys(mat,"rho_kg_m3","cl_m_s","ct_m_s");keys(acq,"duration_us","sample_step_us");
        keys(disc,"coarse_h_mm","fine_h_mm","max_step_us","relative_tolerance","field_nx","field_ny","field_frames");keys(tx,"profile","frequency_mhz","amplitude_pa","cycles","delay_us","phase_deg");eq(tx.get("profile"),"left_edge_uniform");
        double rho=num(mat,"rho_kg_m3"),cl=num(mat,"cl_m_s"),ct=num(mat,"ct_m_s");if(rho!=2700||Math.abs(cl-Math.sqrt(110.3e9/2700))>1e-8||Math.abs(ct-Math.sqrt(26.1e9/2700))>1e-8)throw new IllegalArgumentException("Linear constants do not match material card");
        if(!(s.get("receivers") instanceof List))throw new IllegalArgumentException("Receivers");List<?> rx=(List<?>)s.get("receivers");if(rx.size()<1||rx.size()>8)throw new IllegalArgumentException("Receiver count");
        Map<String,Object> expected=new LinkedHashMap<>();
        put(expected,"schema","workai.comsol.research.case.v1");put(expected,"scenario","guided_plate");put(expected,"boundary","free");
        put(expected,"length_mm",g.get("length_mm"));put(expected,"height_mm",g.get("thickness_mm"));put(expected,"rho",rho);put(expected,"cl",cl);put(expected,"ct",ct);
        for(String k:new String[]{"axisymmetric","conductivity","heat_capacity","expansion","thermal","notch_enabled","notch_x_mm","notch_width_mm","notch_depth_mm","notch_bottom"})put(expected,k,0.0);
        put(expected,"outer_radius_mm",1.0);put(expected,"element_order",4.0);put(expected,"element_budget",40000.0);put(expected,"shot_count",5.0);put(expected,"receiver_count",(double)rx.size());
        for(String k:acq.keySet())put(expected,k,acq.get(k));for(String k:disc.keySet())put(expected,k,disc.get(k));
        put(expected,"bandwidth_estimate_mhz",num(tx,"frequency_mhz")*(1+2/num(tx,"cycles")));
        for(int j=0;j<5;j++){
            String a="tx"+j+"_";put(expected,a+"kind",0.0);put(expected,a+"x_mm",0.0);put(expected,a+"radius_mm",num(g,"thickness_mm")/2);put(expected,a+"direction",0.0);put(expected,a+"frequency_count",1.0);
            put(expected,a+"f0",tx.get("frequency_mhz"));put(expected,a+"amplitude_pa",num(tx,"amplitude_pa")*(j>=3?2:1));for(String k:new String[]{"delay_us","phase_deg","cycles"})put(expected,a+k,tx.get(k));
        }
        Set<String> ids=new HashSet<>();double maxX=0;for(int j=0;j<rx.size();j++){Map<String,Object> r=obj(rx.get(j));keys(r,"id","x_mm");String id=(String)r.get("id");if(!id.matches("[a-zA-Z][a-zA-Z0-9_-]{0,15}")||!ids.add(id))throw new IllegalArgumentException("Receiver ID");maxX=Math.max(maxX,num(r,"x_mm"));put(expected,"rx"+j+"_x_mm",r.get("x_mm"));}
        Map<String,Object> actual=obj(new Json(pj).read());if(!actual.keySet().equals(expected.keySet())||p.size()!=expected.size()+4)throw new IllegalArgumentException("Transport fields differ from scene");
        for(String k:expected.keySet()){
            Object v=expected.get(k);if(v instanceof Double){double x=(Double)v;if(!(actual.get(k) instanceof Double)||Math.abs((Double)actual.get(k)-x)>Math.max(1,Math.abs(x))*1e-12||Math.abs(n(k)-x)>Math.max(1,Math.abs(x))*1e-12)throw new IllegalArgumentException("Numeric scene/transport mismatch: "+k);}else{eq(v,actual.get(k));eq(v,p.getProperty(k));}
        }
        range("length_mm",.1,400);range("height_mm",.01,5);range("tx0_f0",.001,2);range("tx0_cycles",4,100);range("tx0_amplitude_pa",1e5,5e6);
        if(n("tx0_f0")*n("sample_step_us")>.1000000001||n("tx0_f0")*n("max_step_us")>.0250000001)throw new IllegalArgumentException("Second harmonic sampling");
        if(n("tx0_delay_us")+n("tx0_cycles")/n("tx0_f0")>n("duration_us")+1e-9||n("duration_us")>=(2*n("length_mm")-maxX)*1000/cl)throw new IllegalArgumentException("Pulse/reflection time window");
        if(5*rx.size()*(Math.round(n("duration_us")/n("sample_step_us"))+1)>4000||5*i("field_nx")*i("field_ny")*i("field_frames")>4000)throw new IllegalArgumentException("Bounded artifact rows");
    }
    private static String nonlinearWeak(Model m){
        m.param().set("lam","58.1e9");m.param().set("mu","26.1e9");m.param().set("nlA","-345e9");m.param().set("nlB","-124.5e9");m.param().set("nlC","-18.5e9");m.param().set("geom_on","0");m.param().set("mat_on","0");
        m.variable().create("nlvars");m.variable("nlvars").model("comp1");
        String[][] F={{"1+ux","uy","0"},{"vx","1+vy","0"},{"wx","wy","1"}};
        for(int a=0;a<3;a++)for(int b=0;b<3;b++)m.variable("nlvars").set("nf"+a+b,F[a][b]);
        for(int a=0;a<3;a++)for(int b=0;b<3;b++){
            StringJoiner terms=new StringJoiner("+");for(int k=0;k<3;k++)terms.add("nf"+k+a+"*nf"+k+b);
            m.variable("nlvars").set("ne"+a+b,"0.5*("+terms+"-"+(a==b?"1":"0")+")");
        }
        m.variable("nlvars").set("ntr","ne00+ne11+ne22");m.variable("nlvars").set("ntr2","ne00^2+ne11^2+ne22^2+2*(ne01^2+ne02^2+ne12^2)");
        m.variable("nlvars").set("nj","nf00*nf11-nf01*nf10");
        for(int a=0;a<3;a++)for(int b=0;b<3;b++){
            StringJoiner square=new StringJoiner("+");for(int k=0;k<3;k++)square.add("ne"+a+k+"*ne"+k+b);
            m.variable("nlvars").set("ns"+a+b,(a==b?"lam*ntr+mat_on*(nlB*ntr2+nlC*ntr^2)+":"")+"(2*mu+2*mat_on*nlB*ntr)*ne"+a+b+"+mat_on*nlA*("+square+")");
        }
        String[] vars={"u","v","w"},dirs={"x","y"};StringJoiner total=new StringJoiner("+","rho*(utt*test(u)+vtt*test(v)+wtt*test(w))+","");
        for(int a=0;a<3;a++)for(int b=0;b<2;b++){
            StringJoiner sum=new StringJoiner("+");for(int k=0;k<3;k++)sum.add("nf"+a+k+"*ns"+k+b);
            String lin=(a==b?"lam*(ux+vy)+":"")+"mu*("+vars[a]+dirs[b]+(a<2?"+"+vars[b]+dirs[a]:"")+")";
            m.variable("nlvars").set("np"+a+b,"if(geom_on>0.5,"+sum+","+lin+")");total.add("np"+a+b+"*test("+vars[a]+dirs[b]+")");
        }
        return total.toString();
    }
'''
# COMSOL's batch class loader loads the primary .class only. Keep the bounded
# parser in the primary class rather than depending on UltraResearch$Json.class.
start = extra.index('    private static class Json {')
end = extra.index('    @SuppressWarnings', start)
parser = extra[start:end]
parser = parser.replace('    private static class Json {\n', '')
parser = parser[:parser.rfind('    }')]
parser = parser.replace('String s;int at=0;Json(String s){this.s=s;if(s.length()>65536)bad();}', 'private static String s; private static int at;')
for before, after in [('void bad()', 'private static void bad()'), ('void ws()', 'private static void ws()'), ('char take()', 'private static char take()'), ('String str()', 'private static String str()'), ('Object val(int depth)', 'private static Object val(int depth)'), ('Object read()', 'private static Object parseJson(String input)')]:
    parser = parser.replace(before, after)
parser = parser.replace('parseJson(String input){', 'parseJson(String input){s=input;at=0;if(s.length()>65536)bad();')
extra = extra[:start] + parser + extra[end:]
extra = extra.replace('new Json(new String(Files.readAllBytes(Paths.get("request.json")),java.nio.charset.StandardCharsets.UTF_8)).read()', 'parseJson(new String(Files.readAllBytes(Paths.get("request.json")),java.nio.charset.StandardCharsets.UTF_8))')
extra = extra.replace('new Json(sj).read()', 'parseJson(sj)').replace('new Json(pj).read()', 'parseJson(pj)')
src = src.replace('    private static Properties p, receipt;', '    private static Properties p, receipt;\n' + extra)
src = src.replace('m.intRule("quad1").create("order8").order(8);', 'm.intRule("quad1").create("order8").order(nonlinear?20:8);')
src = src.replace('weak(m,"elasticity",2,bulk);', 'weak(m,"elasticity",2,nonlinear?nonlinearWeak(m):bulk);')
src = src.replace('if(hasForce)weak(m,"traction",1,"-if(abs(y-Ytop)<1e-9,"+loads+",0)");', '''if(nonlinear){StringJoiner edge=new StringJoiner("+");for(int k=0;k<5;k++){String a="tx"+k+"_";edge.add("if(active_tx=="+k+","+f(n(a+"amplitude_pa"))+"*"+pulse(a,n(a+"delay_us")*1e-6)+",0)");}weak(m,"traction",1,"-if(abs(x)<1e-9,("+edge+")*test(u),0)");}
        else if(hasForce)weak(m,"traction",1,"-if(abs(y-Ytop)<1e-9,"+loads+",0)");''')
src = src.replace('m.param().set("active_tx",Integer.toString(shot));System.out.println', 'm.param().set("active_tx",Integer.toString(shot));if(nonlinear){m.param().set("geom_on",shot==0?"0":"1");m.param().set("mat_on",shot==2||shot==4?"1":"0");}System.out.println')
src = src.replace('out.flush();field.flush();m.sol(solver).clearSolutionData();', '''if(nonlinear){
                    if(!Arrays.asList(m.result().numerical().tags()).contains("strainGuard"))m.result().numerical().create("strainGuard","MaxSurface");
                    m.result().numerical("strainGuard").set("expr",new String[]{"sqrt(ntr2)","-nj"});m.result().numerical("strainGuard").set("data","dset1");m.result().numerical("strainGuard").selection().all();m.result().numerical("strainGuard").set("solnum",all);
                    double[][] bounds=m.result().numerical("strainGuard").getReal();if(bounds.length!=2)throw new IllegalStateException("Strain guard shape");double strain=0,minJ=Double.POSITIVE_INFINITY;
                    for(double x:bounds[0]){if(!Double.isFinite(x))throw new IllegalStateException("Nonfinite strain");strain=Math.max(strain,x);}for(double x:bounds[1]){if(!Double.isFinite(x))throw new IllegalStateException("Nonfinite Jacobian");minJ=Math.min(minJ,-x);}
                    if(strain>0.001||minJ<=0)throw new IllegalStateException("Outside small elastic strain validation budget");
                    receipt.setProperty(level+"_shot"+shot+"_max_green_strain",f(strain));receipt.setProperty(level+"_shot"+shot+"_min_J",f(minJ));
                }
                out.flush();field.flush();m.sol(solver).clearSolutionData();''')
src = src.replace('level.equals("timecheck")?2.5e-15:1e-14', 'nonlinear?(level.equals("timecheck")?5e-17:2e-16):(level.equals("timecheck")?2.5e-15:1e-14)')
src = src.replace('}validate();if(Files.exists', '}validate();readScene();if(Files.exists')
src = src.replace('receipt.setProperty("method","comsol_research_2d_axisymmetric_weakform_v1")', 'receipt.setProperty("method",nonlinear?"comsol_murnaghan_total_lagrangian_2d_v1":"comsol_research_2d_axisymmetric_weakform_v1")')
src = src.replace('receipt.setProperty("quadrature_order","8")', 'receipt.setProperty("quadrature_order",nonlinear?"20":"8")')
src = src.replace('try(OutputStream out=Files.newOutputStream(Paths.get("completed.properties")))', '''if(nonlinear){receipt.setProperty("constitutive","murnaghan_green_lagrange");receipt.setProperty("energy_definition","murnaghan_invariants_reference_volume");receipt.setProperty("source_profile","left_edge_uniform_nominal_traction");receipt.setProperty("controls",String.join(",",CONTROLS));receipt.setProperty("material_record_sha256","39102235cd2fae341a277e46aa08573ae2294710c79abd104b10491fe5edef0a");receipt.setProperty("lambda_pa","5.81e10");receipt.setProperty("mu_pa","2.61e10");receipt.setProperty("l_pa","-1.43e11");receipt.setProperty("m_pa","-2.97e11");receipt.setProperty("n_pa","-3.45e11");receipt.setProperty("spatial_assumption","plane_strain_z_invariant");}
        try(OutputStream out=Files.newOutputStream(Paths.get("completed.properties")))''')
out = Path(sys.argv[2]);out.parent.mkdir(parents=True, exist_ok=True)
assert 100 < len(src) < 45000
out.write_text(src)
print(hashlib.sha256(src.encode()).hexdigest(), len(src.encode()))
