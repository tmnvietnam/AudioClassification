using NAudio.Wave;
using Python.Runtime;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Threading;

namespace SoundKit
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {
        private Point startPoint;
        private Point endPoint;
        private int startSample;
        private int endSample;
        private DispatcherTimer timer;
        private readonly MainViewModel viewModel;

        private double minX = 85;
        private double maxX = 1362;
        private string datasetDir = string.Empty;
        private string backgroundFolderPath = string.Empty;
        private string objectFolderPath = string.Empty;
        private string modelFilePath = string.Empty;


        private int metatTimePass = 0;
        private int metatTimeTotal = 0;


        private List<MediaPlayer> mediaPlayers = new();
        private List<Tuple<int, int>> audioForTrain = new();

        public string ServicePath = Properties.Settings.Default.ServicePath;
        public string TempPath = Path.Combine(Properties.Settings.Default.ServicePath, "temp");

        public string VirtualEnvPath = Properties.Settings.Default.VirtualEnvPath;

        FixtureWindow Fixture1Window;
        FixtureWindow Fixture2Window;
        FixtureWindow Fixture3Window;



        public MainWindow()
        {

            InitializeComponent();

            viewModel = new MainViewModel();
            this.DataContext = viewModel;

            InitPython();

            LoadMicrophones();

            this.Closed += WindowClosed;

            // Redirect Console output to the TextBox
            var writer = new TextBoxStreamWriter(MyTextBox);
            Console.SetOut(writer);

            // Set fixed position
            this.Top = 0; // Y-coordinate
            this.Left = 0; // X-coordinate

         


        }
        private void WindowClosed(object sender, EventArgs e)
        {
            PythonEngine.EndAllowThreads(0);
            PythonEngine.Shutdown();
        }

        void InitPython()
        {

            var path = Environment.GetEnvironmentVariable("PATH").TrimEnd(';');
            path = string.IsNullOrEmpty(path) ? VirtualEnvPath : path + ";" + VirtualEnvPath;
            Environment.SetEnvironmentVariable("PATH", path, EnvironmentVariableTarget.Process);
            Environment.SetEnvironmentVariable("PATH", VirtualEnvPath, EnvironmentVariableTarget.Process);
            Environment.SetEnvironmentVariable("PYTHONHOME", VirtualEnvPath, EnvironmentVariableTarget.Process);
            Environment.SetEnvironmentVariable("PYTHONPATH", $@"{VirtualEnvPath}\\Lib\\site-packages;{VirtualEnvPath}\\Lib;{ServicePath}", EnvironmentVariableTarget.Process);

            PythonEngine.PythonHome = VirtualEnvPath;
            PythonEngine.PythonPath = PythonEngine.PythonPath + ";" + Environment.GetEnvironmentVariable("PYTHONPATH", EnvironmentVariableTarget.Process);
            PythonEngine.Initialize();
            PythonEngine.BeginAllowThreads();

        }


        private void LoadMicrophones()
        {
            // Create a list to store available microphones
            var microphones = new List<WaveInCapabilities>();

            // Get and list all available input devices (microphones)
            for (int deviceIndex = 0; deviceIndex < WaveIn.DeviceCount; deviceIndex++)
            {
                var deviceInfo = WaveIn.GetCapabilities(deviceIndex);
                microphones.Add(deviceInfo);
              
            }

            // Bind the list to the ComboBox
            MicrophoneComboBox1.ItemsSource = microphones;
            MicrophoneComboBox2.ItemsSource = microphones;
            MicrophoneComboBox3.ItemsSource = microphones;


            // Set the first item (default microphone) as selected if there are any devices
            if (microphones.Count > 0)
            {
                MicrophoneComboBox1.SelectedIndex = 0; // Automatically select the first device (default microphone)
                MicrophoneComboBox2.SelectedIndex = 0; // Automatically select the first device (default microphone)
                MicrophoneComboBox3.SelectedIndex = 0; // Automatically select the first device (default microphone)

            }


        }

        private void OpenFixture1Btn_Click(object sender, RoutedEventArgs e)
        {
            int selectedMicrophoneIndex = MicrophoneComboBox1.SelectedIndex;
            Fixture1Window = new("Fixture 1", selectedMicrophoneIndex, 0);    
            Fixture1Window.Show();

        }

        private void OpenFixture2Btn_Click(object sender, RoutedEventArgs e)
        {
            int selectedMicrophoneIndex = MicrophoneComboBox2.SelectedIndex;
            Fixture2Window = new("Fixture 2", selectedMicrophoneIndex, 1);    
            Fixture2Window.Show();
        }

        private void OpenFixture3Btn_Click(object sender, RoutedEventArgs e)
        {
            int selectedMicrophoneIndex = MicrophoneComboBox3.SelectedIndex;
            Fixture3Window = new("Fixture 3", selectedMicrophoneIndex, 2);
            Fixture3Window.Show();
        }
    }
}