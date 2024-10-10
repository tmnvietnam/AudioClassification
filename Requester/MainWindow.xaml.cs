using NAudio.Wave;
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

        private string pipeName = "MyNamedPipe";

        private int metatTimePass = 0;
        private int metatTimeTotal = 0;

        private bool EnableReTraining = false;


        private List<MediaPlayer> mediaPlayers = new();
        private List<Tuple<int, int>> audioForTrain = new();


        public MainWindow()
        {
            InitializeComponent();

            viewModel = new MainViewModel();
            this.DataContext = viewModel;

            LoadMicrophones();
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

        private void OpenPCB1Btn_Click(object sender, RoutedEventArgs e)
        {
            int selectedMicrophoneIndex = MicrophoneComboBox1.SelectedIndex;

            PCBWindow pcb1Window = new("PCB 1", selectedMicrophoneIndex);
            pcb1Window.Show();
        }

        private void OpenPCB2Btn_Click(object sender, RoutedEventArgs e)
        {
            int selectedMicrophoneIndex = MicrophoneComboBox2.SelectedIndex;

            PCBWindow pcb1Window = new("PCB 2", selectedMicrophoneIndex);
            pcb1Window.Show();
        }

        private void OpenPCB3Btn_Click(object sender, RoutedEventArgs e)
        {
            int selectedMicrophoneIndex = MicrophoneComboBox3.SelectedIndex;

            PCBWindow pcb3Window = new("PCB 3", selectedMicrophoneIndex);
            pcb3Window.Show();
        }


    }
}