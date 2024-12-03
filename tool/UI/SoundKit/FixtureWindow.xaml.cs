using Python.Runtime;
using FontAwesome.Sharp;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using SoundKit.Audio;
using SoundKit.Serial;
using System;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.IO.Ports;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Forms;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.Windows.Threading;
using MessageBox = System.Windows.MessageBox;
using Path = System.IO.Path;
using NAudio.Wave;
using System.Collections.Concurrent;
using System.Drawing;
using Microsoft.Win32;
using OpenFileDialog = System.Windows.Forms.OpenFileDialog;
using Color = System.Windows.Media.Color;
using ColorConverter = System.Windows.Media.ColorConverter;


namespace SoundKit
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class FixtureWindow : System.Windows.Window
    {
        private DispatcherTimer timer;

        public string ServicePath = Properties.Settings.Default.ServicePath;
        public string TempPath = string.Empty;

        private readonly FixtureViewModel viewModel;

        private string datasetDir = string.Empty;
        private string ngFolderPath = string.Empty;
        private string okFolderPath = string.Empty;
        private string modelFilePath = string.Empty;

        public int NumberConnector = 2;


        private int metatTimePass = 0;
        private int metatTimeTotal = 0;

        private bool EnableTraining = false;

        private List<MediaPlayer> mediaPlayers = new List<MediaPlayer>();
        private List<Tuple<int, int>> audioForTrain = new List<Tuple<int, int>>();

        public List<Device> SerialConnections = new List<Device>();
        public Device Scanner;
        public SystemIO SystemIO;

        public int Duration = 1000;      // Duration in seconds, similar to config.DURATION
        static readonly int SampleRate = 22050; // Sample rate, similar to config.SAMPLE_RATE

        static readonly int WindowSize = SampleRate;
        static readonly int StepSize = SampleRate / 10;


        private int idMicrophone = 0;
        private bool resultCheck;

        private dynamic model;
        private dynamic tf;
        private dynamic np;
        private dynamic aiCore;

        TrainingWindow TrainingWindow;
        RealtimeWindow RealtimeWindow;


        public List<short> AudioData = new List<short>();

        private Queue<short[]> audioQueue = new();  // Queue of audio samples
        private static bool stopFlag = false;


        private bool _ReadyTest = false;

        public bool ReadyTest
        {
            get { return _ReadyTest; }
            set
            {
                if (_ReadyTest != value)
                {
                    _ReadyTest = value;
                }

            }
        }


        private bool _CylinderFullDown = false;

        public bool CylinderCome
        {
            get { return _CylinderFullDown; }
            set
            {
                if (_CylinderFullDown != value)
                {
                    _CylinderFullDown = value;
                }
            }
        }

        private bool _BarcodeUse = false;

        public bool BarcodeUse
        {
            get { return _BarcodeUse; }
            set
            {
                if (_BarcodeUse != value)
                {
                    _BarcodeUse = value;
                }
            }
        }

        private bool _BarcodeReady = false;

        public bool BarcodeReady
        {
            get { return _BarcodeReady; }
            set
            {
                if (_BarcodeReady != value)
                {
                    _BarcodeReady = value;
                }
            }
        } 

        public void ConsoleWriteLine(string content)
        {
            // Get the current datetime
            DateTime currentTime = DateTime.Now;

            // Format the datetime for the log
            string formattedTime = currentTime.ToString("yyyy-MM-dd HH:mm:ss");

            Console.WriteLine($"[{formattedTime}] {Title}: {content}");
        }

        private void ScannerSerialReceiverHandler(object sender, EventArgs e)
        {

            string code = "";

            code = Scanner.ReadLine();
            Debug.WriteLine($"code:{code}");
            Scanner.SerialPort.DiscardInBuffer();

            if (code.Length > 5)
            {
                BarcodeReady = true;               
            }
            else
            {
                BarcodeReady = false;             
            }
            Task.Delay(800);
        }

        private void ControllerSerialReceiverHandler(object sender, EventArgs e)
        {
            if (SystemIO.SerialPort.IsOpen)
            {
                List<byte> frame = new List<byte>();
                Task.Delay(50).Wait();
                int size = SystemIO.SerialPort.BytesToRead;
                byte[] bytes = new byte[size];
                try
                {
                    SystemIO.SerialPort.Read(bytes, 0, SystemIO.SerialPort.BytesToRead);
                }
                catch (Exception)
                {
                    return;
                }

                if (bytes.Length < 7) return;
                for (int i = 0; i < bytes.Length; i++)
                {
                    byte startByte = bytes[i];
                    if (startByte == Device.Prefix1)
                    {
                        var secondByte = bytes[i + 1];
                        if (secondByte == Device.Prefix2)
                        {
                            frame.Clear();
                            frame.Add(startByte);
                            frame.Add(secondByte);
                            frame.Add(bytes[i + 2]);

                            if (bytes[i + 2] + 3 >= bytes.Length) return;

                            for (int j = i + 3; j <= bytes[i + 2] + 3; j++)
                            {
                                frame.Add(bytes[j]);
                            }
                            try
                            {
                                SystemIO.DataReceived(new byte[] { frame[4], frame[5], frame[6], frame[7] });
                            }
                            catch (Exception ex)
                            {

                            }
                          
                            return;
                           
                        }
                    }
                }
            }
        }

        private void StartRequestHandler(object sender, EventArgs e)
        {
            Dispatcher.Invoke(() =>
            {

                CylinderCome = true;


            });
        }
        private void LoadComPorts()
        {
            string[] ports = SerialPort.GetPortNames();

            comPortComboBox1.ItemsSource = ports;
            comPortComboBox2.ItemsSource = ports;

            // Set the first available port as the default selected item, if available
            if (ports.Length > 0)
            {
                comPortComboBox1.SelectedItem = ports[0];  // Set the default to the first COM port
                comPortComboBox2.SelectedItem = ports[0];  // Set the default to the first COM port           
            }

            Scanner = new Device(Title,connectionStatusDevice1, txStatusDevice1, rxStatusDevice1);
            SystemIO = new SystemIO(Title, connectionStatusDevice2, txStatusDevice2, rxStatusDevice2);

            Scanner.SerialReceiverHandler -= ScannerSerialReceiverHandler;
            SystemIO.SerialReceiverHandler -= ControllerSerialReceiverHandler;


            Scanner.SerialReceiverHandler += ScannerSerialReceiverHandler;
            SystemIO.SerialReceiverHandler += ControllerSerialReceiverHandler;

            SystemIO.OnStartRequest += StartRequestHandler;

            SerialConnections.Add(Scanner);
            SerialConnections.Add(SystemIO);

        }

        void InitPython()
        {
            using (Py.GIL())
            {
                np = Py.Import("numpy");
                tf = Py.Import("tensorflow");
                aiCore = Py.Import("aicore");
            }
        }

        public FixtureWindow(string titleWindow, int idMicrophone, int idWindow)
        {
            InitializeComponent();
            this.Title = titleWindow;

            this.Closed += WindowClosed;

            this.idMicrophone = idMicrophone;
            viewModel = new FixtureViewModel();

            DataContext = viewModel;

            LoadComPorts();

            InitCheckCommunication();

            InitPython();

            // Set fixed position
            this.Top = 0; // Y-coordinate
            this.Left = 640* idWindow; // X-coordinate       

            TempPath = Path.Combine(ServicePath, $"temp{idWindow}");
            if (Directory.Exists(TempPath))
            {
                Directory.Delete(TempPath, true);
            }
            Directory.CreateDirectory(TempPath);

            NumberConnectorText.Content = $"{NumberConnector}";
        }


        private string GetNextFileName(string folderPath)
        {
            int fileNumber = 1;
            string fileName;

            while (true)
            {
                fileName = $"{fileNumber:D4}.wav";
                string fullPath = Path.Combine(folderPath, fileName);

                if (!File.Exists(fullPath))
                {
                    return fileName;
                }

                fileNumber++;
            }
        }


        // Function to count files in a folder
        public int CountFilesInFolder(string folderPath)
        {
            try
            {
                // Get all files in the directory
                string[] files = Directory.GetFiles(folderPath);

                // Return the count of files
                return files.Length;
            }
            catch (Exception ex)
            {
                ConsoleWriteLine($"An error occurred: {ex.Message}");
                return 0;
            }
        }

        // Function to check and create a directory if it doesn't exist
        public void EnsureDirectoryExists(string folderPath)
        {
            if (!Directory.Exists(folderPath))
            {
                try
                {
                    Directory.CreateDirectory(folderPath);
                    ConsoleWriteLine($"Created directory: {folderPath}");
                }
                catch (Exception ex)
                {
                    ConsoleWriteLine($"Error creating directory {folderPath}: {ex.Message}");
                }
            }
            else
            {
                ConsoleWriteLine($"Directory already exists: {folderPath}");
            }
        }

        private void WindowClosed(object sender, EventArgs e)
        {

            SerialConnections.ForEach(connection =>
            {
                connection.SerialPort.Close();
                connection.SerialPort.Dispose();

            });
        }

        private void ResetMediaPlayer()
        {
            MediaPlayerStack.Children.Clear();
            mediaPlayers.ForEach(mediaPlayer => mediaPlayer.Close());
            mediaPlayers.Clear();
            audioForTrain.Clear();
        }

        private TaskCompletionSource<string> _buttonClickedTaskCompletionSource;

        void timer_Tick(object sender, EventArgs e, MediaPlayer mediaPlayer, System.Windows.Controls.Label lblStatus)
        {
            try
            {
                if (mediaPlayer.Source != null)
                    lblStatus.Content = string.Format("{0} / {1}", mediaPlayer.Position.ToString(@"mm\:ss"), mediaPlayer.NaturalDuration.TimeSpan.ToString(@"mm\:ss"));
                else
                    lblStatus.Content = "";
            }
            catch (Exception ex)
            {
                ConsoleWriteLine(ex.ToString());
            }

        }


        private void EnableTraining_Checked(object sender, RoutedEventArgs e)
        {
            EnableTraining = true;
            AddBtn.IsEnabled = true;

        }

        private void UnableTraining_Checked(object sender, RoutedEventArgs e)
        {
            EnableTraining = false;
            AddBtn.IsEnabled = false; ;

        }


        private void ContinueBtn_Click(object sender, RoutedEventArgs e)
        {
            _buttonClickedTaskCompletionSource?.SetResult("ContinueBtn");
            //ContinueBtn.IsEnabled = false;
        }

        private void StopBtn_Click(object sender, RoutedEventArgs e)
        {
            _buttonClickedTaskCompletionSource?.SetResult("StopBtn");
            //StopBtn.IsEnabled = false;
            //AddBtn.IsEnabled = false;
        }

        private void AddToDataset_Click(object sender, RoutedEventArgs e)
        {
           
            if (audioForTrain.Count > 0)
            {
                audioForTrain.ForEach(audio =>
                {
                    if (audio.Item2 == 1)
                    {
                        string fileName = GetNextFileName(okFolderPath);
                        string filePath = Path.Combine(okFolderPath, fileName);
                        File.Copy(Path.Combine(TempPath, $"{audio.Item1}.wav"), filePath);
                    }
                    else
                    {
                        string fileName = GetNextFileName(ngFolderPath);
                        string filePath = Path.Combine(ngFolderPath, fileName);
                        File.Copy(Path.Combine(TempPath, $"{audio.Item1}.wav"), filePath);
                    }

                });

              

            }

        }


        private void btnListen_Click(object sender, RoutedEventArgs e, MediaPlayer mediaPlayer)
        {
            mediaPlayer.Stop();
            mediaPlayer.Play();
        }
        private void btnStop_Click(object sender, RoutedEventArgs e, MediaPlayer mediaPlayer)
        {
            mediaPlayer.Stop();
        }
        private void chkEnable_Checked_NG(object sender, RoutedEventArgs e, int indexMediaPlayer)
        {
            audioForTrain.Add(new Tuple<int, int>(indexMediaPlayer, 0));
        }

        private void chkEnable_Unchecked_NG(object sender, RoutedEventArgs e, int indexMediaPlayer)
        {
            // Find the tuple with the matching indexMediaPlayer value
            var tupleToRemove = audioForTrain.FirstOrDefault(t => t.Item1 == indexMediaPlayer);

            // Remove the tuple if found
            if (tupleToRemove != null)
            {
                audioForTrain.Remove(tupleToRemove);
            }
        }

        private void chkEnable_Checked_OK(object sender, RoutedEventArgs e, int indexMediaPlayer)
        {
            audioForTrain.Add(new Tuple<int, int>(indexMediaPlayer, 1));
        }

        private void chkEnable_Unchecked_OK(object sender, RoutedEventArgs e, int indexMediaPlayer)
        {
            // Find the tuple with the matching indexMediaPlayer value
            var tupleToRemove = audioForTrain.FirstOrDefault(t => t.Item1 == indexMediaPlayer);

            // Remove the tuple if found
            if (tupleToRemove != null)
            {
                audioForTrain.Remove(tupleToRemove);
            }
        }

        private async Task StartTestHandlerAsync()
        {
         

            while (ReadyTest)
            {          
                if (CylinderCome && (BarcodeReady || !BarcodeUse))
                {                  
                    CylinderCome = false;
                    BarcodeReady = false;

                    await Dispatcher.InvokeAsync(async () =>
                    {
                        int timePass = 0;
                        ResultTime.Content = $"{timePass}";
                       
                        ResultFinal.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#23affc"));
                        ResultFinal.Foreground = new SolidColorBrush(Colors.White);
                        ResultFinal.Content = "TESTING";

                        if (File.Exists(modelFilePath) && (!EnableTraining || Path.Exists(datasetDir)))
                        {
                            ResetMediaPlayer();
                            bool resultSound = false;
                            for (int idxSound = 0; idxSound < NumberConnector; idxSound++)
                            {
                                try
                                {
                                    if (await CheckAsync())
                                    {
                                        timePass++;
                                        ResultTime.Content = $"{timePass}";
                                        resultSound = true;

                                        await Task.Delay(1000);
                                        continue;
                                    }
                                    else
                                    {
                                        resultSound = false;
                                        break;
                                    }
                                }
                                catch (Exception ex)
                                {
                                    MessageBox.Show($"An error occurred: {ex.Message}");
                                }
                            }

                            metatTimeTotal++;

                            // Update final result based on whether any sound was successfully predicted
                            if (resultSound)
                            {
                                ResultFinal.Background = new SolidColorBrush(Colors.LawnGreen);
                                ResultFinal.Foreground = new SolidColorBrush(Colors.White);
                                ResultFinal.Content = "OK";
                                metatTimePass++;                 
                            }
                            else
                            {
                                ResultFinal.Background = new SolidColorBrush(Colors.Red);
                                ResultFinal.Foreground = new SolidColorBrush(Colors.White);
                                ResultFinal.Content = "NG";
                            }

                            await Task.Delay(1000);
                            ResultFinal.Background = new SolidColorBrush(Colors.Yellow);
                            ResultFinal.Foreground = new SolidColorBrush(Colors.Black);
                            ResultFinal.Content = "READY";


                            SystemIO.ResetOUT = true ;
                            SystemIO.SendControl();

                            ResultSumary.Content = $"OK: {metatTimePass}     NG: {metatTimeTotal - metatTimePass}";

                            SystemIO.ResetOUT = false;
                            SystemIO.SendControl();
                        }
                        else
                        {
                            if (!Path.Exists(modelFilePath))
                            {
                                MessageBox.Show("Model file not found.");
                            }

                            if (EnableTraining && !Path.Exists(datasetDir))
                            {
                                MessageBox.Show("Dataset folder not found.");
                            }
                        }
                    
                    });

                   
                }

            }
        }

        private async void StartTestBtn_Click(object sender, RoutedEventArgs e)
        {
            ReadyTest = true;
            SystemIO.ResetOUT = false;
            SystemIO.SendControl();

            ResultFinal.Background = new SolidColorBrush(Colors.Yellow);
            ResultFinal.Foreground = new SolidColorBrush(Colors.Black);
            ResultFinal.Content = "READY";
            StopTestBtn.IsEnabled = true;
            StartTestBtn.IsEnabled = false;

            await Task.Run(() => StartTestHandlerAsync());
        }

        private async void StopTestBtn_Click(object sender, RoutedEventArgs e)
        {
            ReadyTest = false;
            SystemIO.ResetOUT = false;
            SystemIO.SendControl();

            ResultFinal.Background = new SolidColorBrush(Colors.Black);
            ResultFinal.Foreground = new SolidColorBrush(Colors.White);
            ResultFinal.Content = "STOP";
            StopTestBtn.IsEnabled = false;
            StartTestBtn.IsEnabled = true;
        }

        private void chkEnable_Checked_NG(object sender, RoutedEventArgs e, MediaPlayer mediaPlayer, int indexMediaPlayer)
        {
            audioForTrain.Add(new Tuple<int, int>(indexMediaPlayer, 0));
        }

        private void chkEnable_Unchecked_NG(object sender, RoutedEventArgs e, MediaPlayer mediaPlayer, int indexMediaPlayer)
        {
            // Find the tuple with the matching indexMediaPlayer value
            var tupleToRemove = audioForTrain.FirstOrDefault(t => t.Item1 == indexMediaPlayer);

            // Remove the tuple if found
            if (tupleToRemove != null)
            {
                audioForTrain.Remove(tupleToRemove);
            }
        }

        private void chkEnable_Checked_OK(object sender, RoutedEventArgs e, MediaPlayer mediaPlayer, int indexMediaPlayer)
        {
            audioForTrain.Add(new Tuple<int, int>(indexMediaPlayer, 1));
        }

        private void chkEnable_Unchecked_OK(object sender, RoutedEventArgs e, MediaPlayer mediaPlayer, int indexMediaPlayer)
        {
            // Find the tuple with the matching indexMediaPlayer value
            var tupleToRemove = audioForTrain.FirstOrDefault(t => t.Item1 == indexMediaPlayer);

            // Remove the tuple if found
            if (tupleToRemove != null)
            {
                audioForTrain.Remove(tupleToRemove);
            }
        }

        private void AddMediaPlayer(List<short> audioData, int indexMediaPlayer, OxyColor color)
        {

            WrapPanel wrapPanel = new WrapPanel
            {
                HorizontalAlignment = System.Windows.HorizontalAlignment.Center,
                VerticalAlignment = System.Windows.VerticalAlignment.Center

            };

            System.Windows.Controls.CheckBox chkEnable_NG = new System.Windows.Controls.CheckBox
            {
                Name = $"chkEnable_NG",
                Content = "NG",
                IsChecked = false,
                Foreground = new SolidColorBrush(Colors.Red),
            };

            chkEnable_NG.Checked += (sender, e) => chkEnable_Checked_NG(sender, e, indexMediaPlayer);
            chkEnable_NG.Unchecked += (sender, e) => chkEnable_Unchecked_NG(sender, e, indexMediaPlayer);

            Grid gridEnable_NG = new Grid
            {
                VerticalAlignment = System.Windows.VerticalAlignment.Center,
                Margin = new Thickness(5, 0, 0, 0)
            };
            gridEnable_NG.Children.Add(chkEnable_NG);

            System.Windows.Controls.CheckBox chkEnable_OK = new System.Windows.Controls.CheckBox
            {
                Name = $"chkEnable_OK",
                Content = "OK",
                IsChecked = false,
                Foreground = new SolidColorBrush(Colors.Green)

            };

            chkEnable_OK.Checked += (sender, e) => chkEnable_Checked_OK(sender, e, indexMediaPlayer);
            chkEnable_OK.Unchecked += (sender, e) => chkEnable_Unchecked_OK(sender, e, indexMediaPlayer);

            Grid gridEnable_OK = new Grid
            {
                VerticalAlignment = System.Windows.VerticalAlignment.Center,
                Margin = new Thickness(5, 0, 0, 0)
            };
            gridEnable_OK.Children.Add(chkEnable_OK);


            PlotModel plotModel = new PlotModel { };
            plotModel.Series.Add(new LineSeries { LineStyle = LineStyle.Solid, Color = color });
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom });
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Minimum = -20000, Maximum = 20000 });

            OxyPlot.Wpf.PlotView plotView = new OxyPlot.Wpf.PlotView
            {
                Model = plotModel,
                Width = 500,
                Height = 100,
            };

            wrapPanel.Children.Add(plotView);

            wrapPanel.Children.Add(gridEnable_NG);
            wrapPanel.Children.Add(gridEnable_OK);

            MediaPlayerStack.Children.Add(wrapPanel);

            FixtureViewModel.UpdatePlot(plotModel, audioData.ToArray());
        }

      
        public void SaveToWav(string filePath, List<short> audioData)
        {
            try
            {
                var waveFormat = new WaveFormat(SampleRate, 16, 1);
                using (var writer = new WaveFileWriter(filePath, waveFormat))
                {
                    // Convert List<short> to byte array
                    var byteArray = new byte[audioData.Count * sizeof(short)];
                    Buffer.BlockCopy(audioData.ToArray(), 0, byteArray, 0, byteArray.Length);

                    // Write the byte array to the file
                    writer.Write(byteArray, 0, byteArray.Length);
                }
            }
            catch
            {
                ConsoleWriteLine("Failed to save as WAV file.");
            }
            
        }
     
        private void SlidingWindowDetection()
        {
            audioQueue.Clear();

            int index_window = 0;
            List<short> buffer = new List<short>();  // Buffer to hold incoming samples
            using (Py.GIL()) // Acquire the Global Interpreter Lock
            {
                try
                {
                    while (!stopFlag || audioQueue.Count > 0)
                    {
                        if (audioQueue.Count > 0)
                        {
                            Debug.WriteLine("Dequeued the next chunk of samples");
                            // Dequeue the next chunk of samples (short[] from the queue)
                            short[] newSamples = audioQueue.Dequeue();

                            // Append new samples to the buffer
                            buffer.AddRange(newSamples);

                            // Process the buffer if it has enough samples
                            while (buffer.Count >= WindowSize)
                            {
                                // Extract the current window
                                List<short> currentWindow = buffer.GetRange(0, WindowSize);

                                if (EnableTraining)
                                {
                                    string fileName = Path.Combine(TempPath, $@"{index_window}.wav");
                                    SaveToWav(fileName, currentWindow);
                                }

                                List<float> floatList = new List<float>(); // List to store floats
                                foreach (var sample in currentWindow)
                                {
                                    // Convert back to normalized float value
                                    float normalizedValue = sample / (float)(short.MaxValue); // Use consistent scaling factor
                                    floatList.Add(normalizedValue); // Add normalized float value to the list
                                }

                                dynamic segment = np.array(floatList);
                                dynamic resultPredict = aiCore.predict(segment, model);

                                if ((String)resultPredict == "OK")
                                {
                                    Dispatcher.Invoke(() =>
                                    {
                                        AddMediaPlayer(currentWindow, index_window, OxyColors.Green); // Update UI
                                        MediaPlayerScrollViewer.ScrollToEnd();       // Ensure visibility
                                    });

                                    resultCheck = true;
                                    stopFlag = true;
                                    break;
                                }
                                else
                                {
                                    Dispatcher.Invoke(() =>
                                    {
                                        AddMediaPlayer(currentWindow, index_window, OxyColors.Red); // Update UI
                                        MediaPlayerScrollViewer.ScrollToEnd();       // Ensure visibility
                                    });

                                    resultCheck = false;
                                }
                             
                                index_window++;
                                // Slide the window forward by step_size
                                buffer = buffer.GetRange(StepSize, buffer.Count - StepSize);

                            }

                            if (resultCheck)
                            {
                                break;
                            }
                        }
                        else
                        {
                            Thread.Sleep(100); // Avoid busy-waiting if the queue is empty
                            Debug.WriteLine("the queue is empty");
                        }

                    }
                }

                catch (PythonException ex)
                {
                    Debug.WriteLine($"Python error: {ex.Message}");
                }
            }
        }


        protected void OnDataAvailable(object sender, WaveInEventArgs e)
        {
            try
            {
                int sampleCount = e.BytesRecorded / sizeof(short);
                var samples = new short[sampleCount];

                // Copy the audio data from the byte buffer to the samples array
                Buffer.BlockCopy(e.Buffer, 0, samples, 0, e.BytesRecorded);

                // Lock to ensure thread-safety while modifying shared data structures
                lock (AudioData)
                {
                    // Add samples to the AudioData list and audio queue for processing
                    AudioData.AddRange(samples);
                    audioQueue.Enqueue(samples);
                }
            }
            catch (Exception ex)
            {
                ConsoleWriteLine($"Error in OnDataAvailable: {ex.Message}");
            }
        }

        private void RecordSound()
        {

            using (var waveIn = new WaveInEvent())
            {
                waveIn.DeviceNumber = idMicrophone;
                waveIn.WaveFormat = new WaveFormat(SampleRate, 16, 1); // 16-bit PCM, mono
                waveIn.BufferMilliseconds = 100; // Buffer size

                waveIn.DataAvailable += OnDataAvailable;


                waveIn.StartRecording();

                Dispatcher.Invoke(() =>
                {
                    ConsoleWriteLine("Recording started...");
                });

                DateTime startTime = DateTime.Now;
                while (!stopFlag && (DateTime.Now - startTime).TotalMilliseconds < Duration * 1000)
                {
                    if (resultCheck == true)
                    {
                        stopFlag = true; // Stop if the detection result is "OK"
                    }
                    Thread.Sleep(100);  // Check periodically while recording
                }

                waveIn.StopRecording();
                stopFlag = true;
                Dispatcher.Invoke(() =>
                {
                    ConsoleWriteLine("Recording stopped.");
                });
            }
        }

        public async Task<bool> CheckAsync()
        {
            resultCheck = false;
            stopFlag = false;

            // Run detection and recording in parallel using Task.Run
            var detectTask = Task.Run(() => SlidingWindowDetection());
            var recordTask = Task.Run(() => RecordSound());

            // Wait for both tasks to complete
            await Task.WhenAll(detectTask, recordTask);

            return resultCheck;
        }

  
        private void LoadModelButton_Click(object sender, RoutedEventArgs e)
        {

            // Create a new OpenFileDialog instance
            OpenFileDialog openFileDialog = new OpenFileDialog();

            // Show the dialog and get the result
            bool result = openFileDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK;

            // Check if the user selected a file
            if (result == true)
            {
                // Get the selected file path
                string filePath = openFileDialog.FileName;

                // Get the current user's home directory
                string userHomePath = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

                // Remove the user's home directory from the file path
                string relativePath = filePath.StartsWith(userHomePath, StringComparison.OrdinalIgnoreCase)
                    ? "~\\" + filePath.Substring(userHomePath.Length).TrimStart('\\')
                    : filePath;

                LoadModelFilePath.Text = relativePath;

                modelFilePath = filePath;

                using (Py.GIL())
                {
                    model = tf.keras.models.load_model(modelFilePath);
                    model.summary();
                }
            }

        }


        private void InitCheckCommunication()
        {
            try
            {
                Scanner.SerialPort.ReadTimeout = 1000;
                Scanner.SerialPort.NewLine = "\r";

                Scanner.CheckCommunication("COM20");
                SystemIO.CheckCommunication("COM13");

                Scanner.SerialReceiverHandler -= ScannerSerialReceiverHandler;
                SystemIO.SerialReceiverHandler -= ControllerSerialReceiverHandler;

                Scanner.SerialReceiverHandler += ScannerSerialReceiverHandler;
                SystemIO.SerialReceiverHandler += ControllerSerialReceiverHandler;

            }
            catch (Exception exception)
            {
                ConsoleWriteLine(exception.Message);
            }
        }

        private void CheckCommunication_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                Scanner.SerialPort.ReadTimeout = 1000;
                Scanner.SerialPort.NewLine = "\r";

                Scanner.CheckCommunication((string)comPortComboBox1.SelectedItem);
                SystemIO.CheckCommunication((string)comPortComboBox2.SelectedItem);

                Scanner.SerialReceiverHandler -= ScannerSerialReceiverHandler;
                SystemIO.SerialReceiverHandler -= ControllerSerialReceiverHandler;

                Scanner.SerialReceiverHandler += ScannerSerialReceiverHandler;
                SystemIO.SerialReceiverHandler += ControllerSerialReceiverHandler;


            }
            catch (Exception exception)
            {
                ConsoleWriteLine(exception.Message);
            }
        }

        private void ScanCode(object sender, RoutedEventArgs e)
        {
            BarcodeReady = true;
           
        }

        private void OpenDatasetButton_Click(object sender, RoutedEventArgs e)
        {
            using (var folderDialog = new FolderBrowserDialog())
            {
                folderDialog.SelectedPath = @"C:\dataset";

                if (folderDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    string selectedPath = folderDialog.SelectedPath;


                    // Get the current user's home directory
                    string userHomePath = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

                    // Remove the user's home directory from the file path
                    string relativePath = selectedPath.StartsWith(userHomePath, StringComparison.OrdinalIgnoreCase)
                        ? "~\\" +selectedPath.Substring(userHomePath.Length).TrimStart('\\')
                        : selectedPath;

                    DatasetPath.Text = relativePath;
                    datasetDir = selectedPath;


                     
                    if(TrainingWindow!=null)
                    {
                        TrainingWindow.DatasetDir = datasetDir;
                        TrainingWindow.RefreshUIDataset();
                    }
                }
            }
        }

        private void OpenTrainingWindow_Click(object sender, RoutedEventArgs e)
        {
            TrainingWindow = new(Title, idMicrophone, datasetDir);
            TrainingWindow.Show();
        }

        private void OpenRealtimeWindow_Click(object sender, RoutedEventArgs e)
        {
            RealtimeWindow = new(Title, idMicrophone);
            RealtimeWindow.Show();
        }

        private void EnableScanner_Checked(object sender, RoutedEventArgs e)
        {
            BarcodeUse = true;
            ScanCodeBtn.IsEnabled = true;
        }

        private void UnableScanner_Checked(object sender, RoutedEventArgs e)
        {
            BarcodeUse = false;
            ScanCodeBtn.IsEnabled = false;
        }
    }
}