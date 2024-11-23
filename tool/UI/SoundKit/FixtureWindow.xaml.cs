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


namespace SoundKit
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class FixtureWindow : System.Windows.Window
    {
        private DispatcherTimer timer;

        public string ServicePath = Properties.Settings.Default.ServicePath;
        public string TempPath = Path.Combine(Properties.Settings.Default.ServicePath, "temp");

        private readonly FixtureViewModel viewModel;

        private string datasetDir = string.Empty;
        private string ngFolderPath = string.Empty;
        private string okFolderPath = string.Empty;
        private string modelFilePath = string.Empty;


        private int metatTimePass = 0;
        private int metatTimeTotal = 0;

        private bool EnableTraining = false;

        private List<MediaPlayer> mediaPlayers = new List<MediaPlayer>();
        private List<Tuple<int, int>> audioForTrain = new List<Tuple<int, int>>();

        public List<Device> SerialConnections = new List<Device>();
        public Device Scanner;
        public SystemIO SystemIO;

        public int Duration = 100;      // Duration in seconds, similar to config.DURATION
        static readonly int SampleRate = 22050; // Sample rate, similar to config.SAMPLE_RATE

        //static readonly int Duration = 24*3600*7;      // Duration in seconds, similar to config.DURATION
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

        private void ScannerSerialReceiverHandler(object sender, EventArgs e)
        {

            string code = "";

            code = Scanner.ReadLine();
            Trace.WriteLine($"code:{code}");
            Scanner.SerialPort.DiscardInBuffer();

            if (code.Length > 5)
            {
                BarcodeReady = true;

                SystemIO.Lock = true;
                SystemIO.SendControl();
            }
            else
            {
                BarcodeReady = false;
                SystemIO.Lock = false;
                SystemIO.SendControl();
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
                                SystemIO.DataOuput(new byte[] { frame[4], frame[5], frame[6], frame[7] });
                            }
                            catch (Exception ex)
                            {

                            }
                            {
                                Console.Write("SYS INPUT:");
                                foreach (var item in frame)
                                {
                                    Console.Write(item.ToString("X2") + " ");
                                }
                                Console.WriteLine(" ");
                                return;
                            }
                        }
                    }
                }
            }
        }

        private void StartRequestHandler(object sender, EventArgs e)
        {
            Dispatcher.Invoke(() =>
            {
                //ReadyTest = true;
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

            Scanner = new Device(connectionStatusDevice1, txStatusDevice1, rxStatusDevice1);
            SystemIO = new SystemIO(connectionStatusDevice2, txStatusDevice2, rxStatusDevice2);

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

        public FixtureWindow(string titleWindow, int idMicrophone)
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
                Console.WriteLine($"An error occurred: {ex.Message}");
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
                    Console.WriteLine($"Created directory: {folderPath}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error creating directory {folderPath}: {ex.Message}");
                }
            }
            else
            {
                Console.WriteLine($"Directory already exists: {folderPath}");
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
                Console.WriteLine(ex.ToString());
            }

        }


        private void EnableTraining_Checked(object sender, RoutedEventArgs e)
        {
            EnableTraining = true;
            //Duration = 100;
            SaveBtn.IsEnabled = true;

        }

        private void UnableTraining_Checked(object sender, RoutedEventArgs e)
        {
            EnableTraining = false;
            //Duration = 24 * 3600 * 7;
            SaveBtn.IsEnabled = false; ;

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
            //SaveBtn.IsEnabled = false;
        }

        private void SaveToDataset_Click(object sender, RoutedEventArgs e)
        {
            //_buttonClickedTaskCompletionSource?.SetResult("SaveBtn");
            //SaveBtn.IsEnabled = false;
            //StopBtn.IsEnabled = false;
            //ContinueBtn.IsEnabled = true;
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

                //string datasetPath = datasetDir.Replace("\\", "/");
                //await TrainAsync(datasetPath);
                //File.Copy(Path.Combine(viewModel.BackendPath, "model.h5"), modelFilePath, overwrite: true);

                //NumberNGDataSet.Text = $"NG sound: {CountFilesInFolder(ngFolderPath)} files";
                //NumberOKDataSet.Text = $"OK sound: {CountFilesInFolder(okFolderPath)} files";

            }

        }

        private Task<string> WaitForBtnClick()
        {
            _buttonClickedTaskCompletionSource = new TaskCompletionSource<string>();

            //StopBtn.IsEnabled = true;
            //SaveBtn.IsEnabled = true;
            //ContinueBtn.IsEnabled = true;

            // Return a task that completes when either button is clicked
            return _buttonClickedTaskCompletionSource.Task;
        }

        private Task<string> WaitForContinueBtnClick()
        {
            _buttonClickedTaskCompletionSource = new TaskCompletionSource<string>();

            //ContinueBtn.IsEnabled = true;

            // Return a task that completes when either button is clicked
            return _buttonClickedTaskCompletionSource.Task;
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
                if (BarcodeReady)
                {
                    await Dispatcher.InvokeAsync(async () =>
                    {
                        int timePass = 0;
                        ResultTime.Content = $"{timePass}/3";

                        ResultFinal.Background = new SolidColorBrush(Colors.WhiteSmoke);
                        ResultFinal.Foreground = new SolidColorBrush(Colors.Black);
                        ResultFinal.Content = "TESTING";

                        if (File.Exists(modelFilePath) && (!EnableTraining || Path.Exists(datasetDir)))
                        {
                            ResetMediaPlayer();
                            bool resultSound = false;
                            for (int idxSound = 0; idxSound < 3; idxSound++)
                            {
                                try
                                {
                                    if (await CheckAsync())
                                    {
                                        timePass++;
                                        ResultTime.Content = $"{timePass}/3";
                                        resultSound = true;
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

                            SystemIO.Lock = false;
                            SystemIO.SendControl();

                            ResultSumary.Content = $"OK: {metatTimePass}     NG: {metatTimeTotal - metatTimePass}";
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

                    BarcodeReady = false;
                }
            }
        }

        private async void StartTest(object sender, RoutedEventArgs e)
        {
            ReadyTest = true;
            StartTestBtn.IsEnabled = false;
            StopTestBtn.IsEnabled = true;

            await Task.Run(() => StartTestHandlerAsync());

        }
        private async void StopTest(object sender, RoutedEventArgs e)
        {
            ReadyTest = false;

            StartTestBtn.IsEnabled = true;
            StopTestBtn.IsEnabled = false;
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

        //static List<float> ConvertByteArrayToFloatList(List<float> byteArray)
        //{
        //    int byteCount = byteArray.Count / 2; // 16-bit samples, so divide by 2 (2 bytes per sample)
        //    float[] floatArray = new float[byteCount];

        //    for (int i = 0; i < byteCount; i++)
        //    {
        //        short sample = BitConverter.ToInt16(byteArray.ToArray(), i * 2);
        //        // Convert the 16-bit PCM sample to a float between -1 and 1
        //        floatArray[i] = sample / (float)short.MaxValue;
        //    }

        //    return floatArray.ToList();
        //}       

        public string Predict(int index, string modelPath = "")
        {
            string resultPart = string.Empty;
            try
            {
                // Create a NamedPipeClientStream
                using (var pipeClient = new NamedPipeClientStream(".", "TensorflowService", PipeDirection.InOut, PipeOptions.None))
                {
                    // Connect to the pipe
                    pipeClient.Connect();

                    // Write to the pipe
                    string message = $"predict@{index}@{modelPath}";
                    byte[] messageBytes = Encoding.UTF8.GetBytes(message);
                    pipeClient.Write(messageBytes, 0, messageBytes.Length);

                    // Read the response
                    byte[] buffer = new byte[64 * 1024]; // 64 KB buffer
                    int bytesRead = pipeClient.Read(buffer, 0, buffer.Length);
                    string response = Encoding.UTF8.GetString(buffer, 0, bytesRead);

                    // Parse and format the accuracy value
                    if (response.Contains("response:"))
                    {
                        try
                        {
                            // Extract the accuracy value from the response
                            resultPart = response.Split(new[] { "response:" }, StringSplitOptions.None)[1].Trim();


                        }
                        catch (Exception ex)
                        {
                            // Handle exception when parsing accuracy
                        }
                    }
                    else
                    {
                        // Handle case where no accuracy data is received                      
                    }

                }
            }
            catch (IOException ex)
            {
                Console.WriteLine($"I/O error: {ex.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Unexpected error: {ex.Message}");
            }

            return resultPart;
        }

        // Method to save the audio buffer as a .wav file
        public static void SaveToWav(string filePath, List<short> audioData)
        {
            // Convert the List<float> to a byte array
            byte[] byteArray = FloatListToByteArray(audioData);

            // Create a WaveFileWriter to save the audio data as a .wav file
            using (var writer = new WaveFileWriter(filePath, new WaveFormat(SampleRate, 16, 1)))  // 16-bit PCM, mono
            {
                writer.Write(byteArray, 0, byteArray.Length);
            }

            Console.WriteLine($"Audio saved to {filePath}");
        }

        // Helper method to convert a List<float> to a byte array
        private static byte[] FloatListToByteArray(List<short> audioData)
        {
            // Convert each float value to a byte (16-bit PCM)
            List<byte> byteList = new List<byte>();
            foreach (var sample in audioData)
            {
                short pcmValue = (short)(sample * short.MaxValue); // Normalize and convert to 16-bit PCM
                byte[] bytes = BitConverter.GetBytes(pcmValue);
                byteList.AddRange(bytes);
            }
            return byteList.ToArray();
        }

        private void SlidingWindowDetection()
        {
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

                                string fileName = Path.Combine(TempPath, $@"{index_window}.wav");
                                SaveToWav(fileName, currentWindow);

                                List<float> floatList = new List<float>(); // List to store floats
                                foreach (var sample in currentWindow)
                                {
                                    short pcmValue = (short)(sample * short.MaxValue); // Normalize and convert to 16-bit PCM
                                    float normalizedValue = pcmValue / (float)short.MaxValue; // Normalize to range [-1.0, 1.0]
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

                                #region
                                ////////////////////////////////////////////////
                                //Dispatcher.Invoke(() =>
                                //{
                                //    AddMediaPlayer(currentWindow, OxyColors.Blue); // Update UI
                                //    MediaPlayerScrollViewer.ScrollToEnd();       // Ensure visibility
                                //});
                                ////////////////////////////////////////////////\
                                //if using pipe



                                //string resultPredict = Predict(index_window);

                                //if ((String)resultPredict == "OK")
                                //{
                                //    Dispatcher.Invoke(() =>
                                //    {
                                //        AddMediaPlayer(currentWindow, OxyColors.Green); // Update UI
                                //        MediaPlayerScrollViewer.ScrollToEnd();       // Ensure visibility
                                //    });

                                //    resultCheck = true;
                                //    stopFlag = true;
                                //    break;

                                //}
                                //else
                                //{
                                //    Dispatcher.Invoke(() =>
                                //    {
                                //        AddMediaPlayer(currentWindow, OxyColors.Red); // Update UI
                                //        MediaPlayerScrollViewer.ScrollToEnd();       // Ensure visibility
                                //    });

                                //    resultCheck = false;
                                //}


                                #endregion
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
                Console.WriteLine($"Error in OnDataAvailable: {ex.Message}");
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
                Debug.WriteLine("Recording started...");

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
                Debug.WriteLine("Recording stopped.");
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

                LoadModelFilePath.Text = filePath;

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
                SystemIO.CheckCommunication("COM23");

                Scanner.SerialReceiverHandler -= ScannerSerialReceiverHandler;
                SystemIO.SerialReceiverHandler -= ControllerSerialReceiverHandler;

                Scanner.SerialReceiverHandler += ScannerSerialReceiverHandler;
                SystemIO.SerialReceiverHandler += ControllerSerialReceiverHandler;

            }
            catch (Exception exception)
            {
                Console.WriteLine(exception.Message);
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
                Console.WriteLine(exception.Message);
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

                    DatasetPath.Text = selectedPath;
                    datasetDir = DatasetPath.Text;

                    ngFolderPath = Path.Combine(datasetDir, "NG");
                    okFolderPath = Path.Combine(datasetDir, "OK");

                    EnsureDirectoryExists(ngFolderPath);
                    EnsureDirectoryExists(okFolderPath);


                }
            }
        }

        private void OpenTrainingWindow_Click(object sender, RoutedEventArgs e)
        {
            TrainingWindow = new(idMicrophone, datasetDir);
            TrainingWindow.Show();
        }

        private void OpenRealtimeWindow_Click(object sender, RoutedEventArgs e)
        {
            RealtimeWindow = new(idMicrophone);
            RealtimeWindow.Show();
        }
    }
}