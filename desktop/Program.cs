// WorkAI desktop launcher: a native window for the offline workbench.
// It starts the bundled Python computation server (runtime\python.exe -m offline.app), shows the workbench in
// Microsoft Edge WebView2, and stops the server (and any computation worker) when the window closes.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

namespace WorkAI.Desktop
{
    internal static class Program
    {
        public const string Version = "1.0.0";
        public const string Title = "WorkAI 超声导波仿真";

        [STAThread]
        private static int Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.SetUnhandledExceptionMode(UnhandledExceptionMode.CatchException);
            Application.ThreadException += (s, e) => MessageBox.Show("程序遇到错误：" + e.Exception.Message, Title, MessageBoxButtons.OK, MessageBoxIcon.Error);
            Options options;
            try { options = Options.Parse(args); }
            catch (ArgumentException error)
            {
                MessageBox.Show(error.Message, Title, MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 2;
            }
            string root = AppDomain.CurrentDomain.BaseDirectory;
            using (var mutex = new Mutex(true, "Local\\WorkAI.Desktop." + StableHash(root), out bool created))
            {
                if (!created && options.Smoke == null)
                {
                    MessageBox.Show("WorkAI 已经在运行，请使用已打开的窗口。", Title, MessageBoxButtons.OK, MessageBoxIcon.Information);
                    return 0;
                }
                var form = new MainForm(root, options);
                Application.Run(form);
                return form.ExitCode;
            }
        }

        private static string StableHash(string text)
        {
            uint hash = 2166136261;
            foreach (char c in text.ToLowerInvariant()) { hash ^= c; hash *= 16777619; }
            return hash.ToString("x8");
        }

        public static Icon LoadIcon(int size = 0)
        {
            using (Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("WorkAI.Desktop.workai.ico"))
            {
                if (stream == null) return SystemIcons.Application;
                var icon = new Icon(stream);
                return size > 0 ? new Icon(icon, size, size) : icon;
            }
        }
    }

    internal sealed class Options
    {
        public string Smoke;     // --smoke <png>: start, load the workbench, save a screenshot and a JSON check, exit
        public string Data;      // --data <directory>: data directory (default: <program>\data, else %LOCALAPPDATA%\WorkAI\data)
        public bool DevTools;    // --devtools: allow the browser developer tools

        public static Options Parse(string[] args)
        {
            var options = new Options();
            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--smoke": options.Smoke = Path.GetFullPath(Next(args, ref i)); break;
                    case "--data": options.Data = Path.GetFullPath(Next(args, ref i)); break;
                    case "--devtools": options.DevTools = true; break;
                    default: throw new ArgumentException("未知参数：" + args[i] + "\n用法：WorkAI.exe [--data 目录] [--devtools] [--smoke 截图.png]");
                }
            }
            return options;
        }

        private static string Next(string[] args, ref int i)
        {
            if (i + 1 >= args.Length) throw new ArgumentException("参数 " + args[i] + " 需要一个值。");
            return args[++i];
        }
    }

    /// <summary>Kill-on-close job object: computation workers started by the server end with the launcher.</summary>
    internal static class JobObject
    {
        [StructLayout(LayoutKind.Sequential)]
        private struct BasicLimits
        {
            public long PerProcessUserTimeLimit, PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass, SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct IoCounters { public ulong ReadOperations, WriteOperations, OtherOperations, ReadBytes, WriteBytes, OtherBytes; }

        [StructLayout(LayoutKind.Sequential)]
        private struct ExtendedLimits
        {
            public BasicLimits Basic;
            public IoCounters Io;
            public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool SetInformationJobObject(IntPtr job, int infoClass, ref ExtendedLimits info, uint length);
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool CloseHandle(IntPtr handle);

        public static IntPtr Create()
        {
            IntPtr job = CreateJobObject(IntPtr.Zero, null);
            if (job == IntPtr.Zero) return IntPtr.Zero;
            var limits = new ExtendedLimits();
            limits.Basic.LimitFlags = 0x2000;   // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            SetInformationJobObject(job, 9, ref limits, (uint)Marshal.SizeOf(typeof(ExtendedLimits)));
            return job;
        }

        public static void Assign(IntPtr job, IntPtr process) { if (job != IntPtr.Zero) AssignProcessToJobObject(job, process); }
        public static void Close(IntPtr job) { if (job != IntPtr.Zero) CloseHandle(job); }
    }

    /// <summary>The bundled Python computation server.</summary>
    internal sealed class Backend : IDisposable
    {
        private readonly string root, data, logPath;
        private readonly object gate = new object();
        private readonly LinkedList<string> tail = new LinkedList<string>();
        private readonly TaskCompletionSource<string> address = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        private Process process;
        private IntPtr job;

        public Backend(string root, string data)
        {
            this.root = root;
            this.data = data;
            Directory.CreateDirectory(Path.Combine(data, "logs"));
            logPath = Path.Combine(data, "logs", "server.log");
        }

        public string Url => address.Task.Status == TaskStatus.RanToCompletion ? address.Task.Result : null;

        public Task<string> StartAsync()
        {
            string python = Path.Combine(root, "runtime", "python.exe");
            if (!File.Exists(python))
                throw new FileNotFoundException("软件包不完整：缺少 runtime\\python.exe。请把压缩包完整解压到一个可写目录后再运行。");
            var info = new ProcessStartInfo(python, "-X utf8 -m offline.app --no-browser --data \"" + data + "\"")
            {
                WorkingDirectory = root,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
            };
            info.EnvironmentVariables["PYTHONUTF8"] = "1";
            info.EnvironmentVariables["OPENBLAS_NUM_THREADS"] = "1";
            info.EnvironmentVariables["OMP_NUM_THREADS"] = "1";
            process = new Process { StartInfo = info, EnableRaisingEvents = true };
            process.OutputDataReceived += (s, e) => Line(e.Data);
            process.ErrorDataReceived += (s, e) => Line(e.Data);
            // give the output readers a moment so the report carries the server's last lines
            process.Exited += (s, e) => Task.Delay(400).ContinueWith(_ => address.TrySetException(new InvalidOperationException("计算内核已退出。\n\n" + Tail())));
            Line("[launcher] WorkAI " + Program.Version + " 启动计算内核：" + python);
            process.Start();
            job = JobObject.Create();
            JobObject.Assign(job, process.Handle);
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            return address.Task;
        }

        private void Line(string text)
        {
            if (text == null) return;
            lock (gate)
            {
                tail.AddLast(text);
                while (tail.Count > 14) tail.RemoveFirst();
                try { File.AppendAllText(logPath, DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss ") + text + Environment.NewLine, Encoding.UTF8); }
                catch (IOException) { }
            }
            Match match = Regex.Match(text, @"http://127\.0\.0\.1:[0-9]+");
            if (match.Success) address.TrySetResult(match.Value);
        }

        public string Tail() { lock (gate) return string.Join(Environment.NewLine, tail); }
        public string LogPath => logPath;

        /// <summary>Ask the server to stop (it pauses a running computation at its checkpoint), then make sure it is gone.</summary>
        public void Shutdown()
        {
            if (process == null) return;
            try
            {
                if (!process.HasExited && Url != null)
                {
                    using (var web = new WebClient { Encoding = Encoding.UTF8, Proxy = null })
                    {
                        string info = web.DownloadString(Url + "/api/info");
                        Match token = Regex.Match(info, "\"token\"\\s*:\\s*\"([0-9a-f]+)\"");
                        if (token.Success)
                        {
                            web.Headers[HttpRequestHeader.ContentType] = "application/json";
                            web.Headers["X-WorkAI-Token"] = token.Groups[1].Value;
                            web.UploadString(Url + "/api/shutdown", "POST", "{}");
                        }
                    }
                }
                if (!process.HasExited && !process.WaitForExit(12000)) process.Kill();
            }
            catch (Exception)
            {
                try { if (!process.HasExited) process.Kill(); } catch (Exception) { }
            }
        }

        public void Dispose()
        {
            Shutdown();
            JobObject.Close(job);   // ends anything still attached to the job (a computation worker)
            job = IntPtr.Zero;
            process?.Dispose();
            process = null;
        }
    }

    internal sealed class SplashForm : Form
    {
        private readonly Label status;

        public SplashForm(Icon icon)
        {
            FormBorderStyle = FormBorderStyle.None;
            StartPosition = FormStartPosition.CenterScreen;
            AutoScaleMode = AutoScaleMode.Dpi;
            AutoScaleDimensions = new SizeF(96F, 96F);
            ClientSize = new Size(480, 270);
            BackColor = Color.White;
            ShowInTaskbar = false;
            Icon = icon;
            Text = Program.Title;
            var font = "Microsoft YaHei UI";
            Controls.Add(new Panel { Dock = DockStyle.Top, Height = 6, BackColor = Color.FromArgb(31, 111, 178) });
            Controls.Add(new PictureBox { Image = Program.LoadIcon(64).ToBitmap(), SizeMode = PictureBoxSizeMode.CenterImage, Bounds = new Rectangle(30, 48, 72, 72) });
            Controls.Add(new Label { Text = Program.Title, Font = new Font(font, 17F, FontStyle.Bold), ForeColor = Color.FromArgb(29, 40, 51), AutoSize = true, Location = new Point(114, 54) });
            Controls.Add(new Label { Text = "版本 " + Program.Version + " · 非线性超声导波物理 AI 仿真", Font = new Font(font, 9.5F), ForeColor = Color.FromArgb(93, 107, 120), AutoSize = true, Location = new Point(117, 96) });
            status = new Label { Text = "正在启动…", Font = new Font(font, 9.5F), ForeColor = Color.FromArgb(29, 40, 51), AutoSize = true, Location = new Point(32, 182) };
            Controls.Add(status);
            Controls.Add(new ProgressBar { Style = ProgressBarStyle.Marquee, MarqueeAnimationSpeed = 30, Bounds = new Rectangle(32, 208, 416, 8) });
            Controls.Add(new Label { Text = "本机离线运行，不需要联网 · 首次计算需要编译，约数分钟", Font = new Font(font, 8.5F), ForeColor = Color.FromArgb(135, 148, 160), AutoSize = true, Location = new Point(32, 228) });
            Paint += (s, e) => e.Graphics.DrawRectangle(new Pen(Color.FromArgb(142, 163, 184)), 0, 0, ClientSize.Width - 1, ClientSize.Height - 1);
        }

        public void Status(string text) { status.Text = text; status.Refresh(); }
    }

    internal sealed class MainForm : Form
    {
        private readonly string root;
        private readonly Options options;
        private readonly string data;
        private readonly WebView2 web;
        private Backend backend;
        private SplashForm splash;
        private string address;
        private bool allowClose, closing, shown;

        public int ExitCode { get; private set; }

        public MainForm(string root, Options options)
        {
            this.root = root;
            this.options = options;
            AutoScaleMode = AutoScaleMode.Dpi;
            AutoScaleDimensions = new SizeF(96F, 96F);
            Text = Program.Title + " " + Program.Version;
            Icon = Program.LoadIcon();
            MinimumSize = new Size(1100, 720);
            StartPosition = FormStartPosition.CenterScreen;
            Size = new Size(1600, 950);
            BackColor = Color.FromArgb(233, 236, 240);
            data = ResolveDataDirectory();
            web = new WebView2 { Dock = DockStyle.Fill, DefaultBackgroundColor = BackColor };
            Controls.Add(web);
            if (options.Smoke == null) WindowState = FormWindowState.Maximized;
        }

        private string ResolveDataDirectory()
        {
            if (options.Data != null) return options.Data;
            string local = Path.Combine(root, "data");
            try
            {
                Directory.CreateDirectory(local);
                string probe = Path.Combine(local, ".write-test");
                File.WriteAllText(probe, "ok");
                File.Delete(probe);
                return local;
            }
            catch (Exception)
            {
                return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "WorkAI", "data");
            }
        }

        protected override async void OnShown(EventArgs e)
        {
            base.OnShown(e);
            splash = new SplashForm(Icon) { TopMost = true };
            if (options.Smoke == null) splash.Show();
            try
            {
                splash.Status("正在启动计算内核…");
                backend = new Backend(root, data);
                address = await WithTimeout(backend.StartAsync(), 120000, "计算内核启动超时。");
                splash.Status("正在打开工作台…");
                var environment = await CoreWebView2Environment.CreateAsync(null, Path.Combine(data, "webview2"));
                await web.EnsureCoreWebView2Async(environment);
                Configure();
                web.CoreWebView2.Navigate(address + "/");
            }
            catch (WebView2RuntimeNotFoundException)
            {
                Fallback();
            }
            catch (Exception error)
            {
                Fail(error.Message);
            }
        }

        private static async Task<T> WithTimeout<T>(Task<T> task, int milliseconds, string message)
        {
            if (await Task.WhenAny(task, Task.Delay(milliseconds)) != task) throw new TimeoutException(message);
            return await task;
        }

        private bool Local(string uri) =>
            uri.StartsWith(address + "/", StringComparison.OrdinalIgnoreCase) || uri == address ||
            uri.StartsWith("blob:", StringComparison.OrdinalIgnoreCase) || uri.StartsWith("data:", StringComparison.OrdinalIgnoreCase) || uri == "about:blank";

        private void Configure()
        {
            CoreWebView2 core = web.CoreWebView2;
            CoreWebView2Settings settings = core.Settings;
            settings.AreDevToolsEnabled = options.DevTools;
            settings.AreBrowserAcceleratorKeysEnabled = options.DevTools;   // F5 / Ctrl+F etc. off; the page's own shortcuts still work
            settings.IsStatusBarEnabled = false;
            settings.IsPasswordAutosaveEnabled = false;
            settings.IsGeneralAutofillEnabled = false;
            core.DocumentTitleChanged += (s, e) => Text = string.IsNullOrWhiteSpace(core.DocumentTitle) ? Program.Title + " " + Program.Version : core.DocumentTitle;
            core.NewWindowRequested += (s, e) => { e.Handled = true; OpenExternal(e.Uri); };
            core.NavigationStarting += (s, e) => { if (!Local(e.Uri)) { e.Cancel = true; OpenExternal(e.Uri); } };
            core.WebMessageReceived += (s, e) => { if (e.WebMessageAsJson.Contains("\"exit\"")) Close(); };
            core.DownloadStarting += OnDownload;
            core.NavigationCompleted += OnNavigationCompleted;
            core.ProcessFailed += (s, e) =>
            {
                if (e.ProcessFailedKind == CoreWebView2ProcessFailedKind.RenderProcessExited || e.ProcessFailedKind == CoreWebView2ProcessFailedKind.RenderProcessUnresponsive)
                    core.Reload();
            };
        }

        /// <summary>Native "save as" dialog for model files, screenshots and result archives.</summary>
        private void OnDownload(object sender, CoreWebView2DownloadStartingEventArgs e)
        {
            if (options.Smoke != null) { e.Cancel = true; return; }
            CoreWebView2Deferral deferral = e.GetDeferral();
            BeginInvoke(new Action(() =>
            {
                try
                {
                    string name = Path.GetFileName(e.ResultFilePath), extension = Path.GetExtension(name).ToLowerInvariant();
                    string filter = extension == ".json" ? "WorkAI 模型 (*.json)|*.json" : extension == ".zip" ? "结果压缩包 (*.zip)|*.zip" : extension == ".png" ? "PNG 图像 (*.png)|*.png" : extension == ".csv" ? "CSV 表格 (*.csv)|*.csv" : "所有文件 (*.*)|*.*";
                    using (var dialog = new SaveFileDialog { FileName = name, InitialDirectory = Path.GetDirectoryName(e.ResultFilePath), Filter = filter, OverwritePrompt = true, Title = "保存" })
                    {
                        if (dialog.ShowDialog(this) == DialogResult.OK) { e.ResultFilePath = dialog.FileName; e.Handled = true; }
                        else e.Cancel = true;
                    }
                }
                finally { deferral.Complete(); }
            }));
        }

        private void OnNavigationCompleted(object sender, CoreWebView2NavigationCompletedEventArgs e)
        {
            if (shown) return;
            if (!e.IsSuccess) { Fail("工作台页面加载失败（" + e.WebErrorStatus + "）。"); return; }
            shown = true;
            splash?.Close();
            splash = null;
            if (options.Smoke == null) Activate();
            else RunSmoke();
        }

        private async void RunSmoke()
        {
            try
            {
                CoreWebView2 core = web.CoreWebView2;
                string ready = "";
                for (int i = 0; i < 240 && ready != "\"1\"" && ready != "\"error\""; i++)
                {
                    await Task.Delay(500);
                    ready = await core.ExecuteScriptAsync("(document.body && document.body.dataset.ready) || ''");
                }
                if (ready != "\"1\"") throw new InvalidOperationException("工作台未就绪：" + ready);
                await Task.Delay(3000);   // let the model check and the 3-D view settle
                string checks = await core.ExecuteScriptAsync(
                    "JSON.stringify({title: document.title, treeRows: document.querySelectorAll('.mb-row').length, ribbonButtons: document.querySelectorAll('.rb-btn').length, " +
                    "status: document.querySelector('#sb-state').textContent, app: document.querySelector('#sb-app').textContent, messages: document.querySelector('#dock-messages').innerText.slice(0, 600)})");
                using (FileStream file = File.Create(options.Smoke))
                    await core.CapturePreviewAsync(CoreWebView2CapturePreviewImageFormat.Png, file);
                File.WriteAllText(Path.ChangeExtension(options.Smoke, ".json"), Regex.Unescape(checks.Trim('"')), new UTF8Encoding(false));
                ExitCode = 0;
            }
            catch (Exception error)
            {
                File.WriteAllText(Path.ChangeExtension(options.Smoke, ".error.txt"), error + Environment.NewLine + Environment.NewLine + backend?.Tail(), new UTF8Encoding(false));
                ExitCode = 1;
            }
            allowClose = true;
            Close();
        }

        private void OpenExternal(string uri)
        {
            if (!(uri.StartsWith("http://", StringComparison.OrdinalIgnoreCase) || uri.StartsWith("https://", StringComparison.OrdinalIgnoreCase))) return;
            try { Process.Start(new ProcessStartInfo(uri) { UseShellExecute = true }); } catch (Exception) { }
        }

        /// <summary>No WebView2 runtime: show the workbench in the default browser and keep a small control window.</summary>
        private void Fallback()
        {
            if (options.Smoke != null) { Fail("未检测到 Microsoft Edge WebView2 运行时。"); return; }
            splash?.Close();
            splash = null;
            Controls.Remove(web);
            MinimumSize = new Size(0, 0);
            WindowState = FormWindowState.Normal;
            ClientSize = new Size(460, 170);
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            var font = new Font("Microsoft YaHei UI", 9.5F);
            Controls.Add(new Label { Text = "未检测到 Microsoft Edge WebView2 运行时（Windows 10/11 通常自带，可从微软官网安装）。\n工作台已在默认浏览器中打开。关闭本窗口即退出 WorkAI。", Font = font, Bounds = new Rectangle(18, 16, 424, 70) });
            var open = new Button { Text = "在浏览器中打开", Font = font, Bounds = new Rectangle(200, 112, 130, 32) };
            var quit = new Button { Text = "退出", Font = font, Bounds = new Rectangle(340, 112, 100, 32) };
            open.Click += (s, e) => OpenExternal(address + "/");
            quit.Click += (s, e) => Close();
            Controls.Add(open);
            Controls.Add(quit);
            OpenExternal(address + "/");
        }

        private void Fail(string message)
        {
            splash?.Close();
            splash = null;
            ExitCode = 1;
            if (options.Smoke != null)
                File.WriteAllText(Path.ChangeExtension(options.Smoke, ".error.txt"), message + Environment.NewLine + Environment.NewLine + backend?.Tail(), new UTF8Encoding(false));
            else
                MessageBox.Show(message + "\n\n运行日志：" + (backend?.LogPath ?? Path.Combine(data, "logs")), Program.Title, MessageBoxButtons.OK, MessageBoxIcon.Error);
            allowClose = true;
            Close();
        }

        protected override async void OnFormClosing(FormClosingEventArgs e)
        {
            if (allowClose || options.Smoke != null || web.CoreWebView2 == null) { base.OnFormClosing(e); return; }
            e.Cancel = true;
            if (closing) return;
            closing = true;
            try
            {
                string busy = await web.CoreWebView2.ExecuteScriptAsync("window.workaiBusy ? window.workaiBusy() : false");
                if (busy == "true" && MessageBox.Show(this, "正在计算。退出会停止计算；5 ms 及以上的长记录已保存的分块，下次打开后可以继续。\n\n确定退出 WorkAI 吗？",
                        Program.Title, MessageBoxButtons.OKCancel, MessageBoxIcon.Warning) != DialogResult.OK)
                {
                    closing = false;
                    return;
                }
            }
            catch (Exception) { }
            allowClose = true;
            Close();
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            Hide();
            splash?.Close();
            backend?.Dispose();
            base.OnFormClosed(e);
        }
    }
}
