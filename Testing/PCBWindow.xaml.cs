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


namespace SoundKit
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class PCBWindow : System.Windows.Window
    {
        private Point startPoint;

        private Point endPoint;

        private int startSample;

        private int endSample;
        private DispatcherTimer timer;


        private readonly PCBViewModel viewModel;

        private double minX = 85;
        private double maxX = 1367;

        private string datasetDir = string.Empty;
        private string ngFolderPath = string.Empty;
        private string okFolderPath = string.Empty;
        private string modelFilePath = string.Empty;

        private string pipeName = "TensorflowService";
        private int idMicrophone = 0;


        private int metatTimePass = 0;
        private int metatTimeTotal = 0;

        private bool EnableReTraining = false;

        private List<MediaPlayer> mediaPlayers = new List<MediaPlayer>();
        private List<Tuple<int, int>> audioForTrain = new List<Tuple<int, int>>();

        public List<Device> SerialConnections = new List<Device>();
        public Device Scanner;
        public SystemIO SystemIO;


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
                if (ReadyTest == false)
                {
                    if (TestingTab.IsSelected)
                    {
                        //ReadyTest = true;
                    }
                    else
                    {
                        //ReadyTest = false;
                    }
                }
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
        public PCBWindow(string titleWindow, int idMIC, string nameService)
        {
            InitializeComponent();

            Title = titleWindow;
            pipeName = nameService;
            idMicrophone = idMIC;

            this.Closed += Window_Closed;


            viewModel = new PCBViewModel(idMicrophone);

            DataContext = viewModel;

            LoadComPorts();

            CheckCommunication_Init();

            //InitConnectionBackEndML();

            viewModel.AudioRecorderTestingPage.ProgressChanged += UpdateTestProgressBar;


            // Set up a timer to update the plot periodically
            timer = new DispatcherTimer
            {
                Interval = TimeSpan.FromMilliseconds(50)
            };
            timer.Tick += viewModel.UpdatePlotRealtime;
            timer.Start();

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

        private void Window_Closed(object sender, EventArgs e)
        {
            //viewModel.AudioRecorderLearningPage.Dispose();
            SerialConnections.ForEach(connection =>
            {
                connection.SerialPort.Close();                
                connection.SerialPort.Dispose();

            });

        }

        private void Rectangle_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (sender is Rectangle rectangle)
            {
                rectangle.CaptureMouse();
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

        private void SaveModelFileButton_Click(object sender, RoutedEventArgs e)
        {
            // Create an instance of SaveFileDialog
            SaveFileDialog saveFileDialog = new SaveFileDialog
            {
                FileName = "model", // Default file name
                DefaultExt = ".h5" // Default file extension
            };

            bool result = saveFileDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK;
            // Process the result
            if (result == true)
            {
                string filePath = saveFileDialog.FileName;

                File.Copy(Path.Combine(viewModel.BackendPath, "model.h5"), filePath, overwrite: true);
                MessageBox.Show("Model file saved successfully!", "Success", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }


        public void InitConnectionBackEndML()
        {
            try
            {
                // Create a NamedPipeClientStream
                using (var pipeClient = new NamedPipeClientStream(".", pipeName, PipeDirection.InOut, PipeOptions.None))
                {
                    // Connect to the pipe
                    pipeClient.Connect();

                    // Write to the pipe
                    string message = $"init@{idMicrophone}";
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
                            string resultPart = response.Split(new[] { "response:" }, StringSplitOptions.None)[1].Trim();
                            viewModel.BackendPath = resultPart;
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

        }

        private void UpdateTestProgressBar(double progress)
        {
            Dispatcher.Invoke(() => TestRecordingProgressBar.Value = progress);
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
            EnableReTraining = true;
        }

        private void UnableTraining_Checked(object sender, RoutedEventArgs e)
        {
            EnableReTraining = false;
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
            //TrainBtn.IsEnabled = false;
        }

        private void TrainBtn_Click(object sender, RoutedEventArgs e)
        {
            _buttonClickedTaskCompletionSource?.SetResult("TrainBtn");
            //TrainBtn.IsEnabled = false;
            //StopBtn.IsEnabled = false;
            //ContinueBtn.IsEnabled = true;

        }

        private Task<string> WaitForBtnClick()
        {
            _buttonClickedTaskCompletionSource = new TaskCompletionSource<string>();

            //StopBtn.IsEnabled = true;
            //TrainBtn.IsEnabled = true;
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

        private async Task StartTestHandler()
        {
            while (ReadyTest)
            {
                if (BarcodeReady)
                {
                    await Dispatcher.Invoke(async () =>
                    {
                        //int maxRetryTime = 60;
                        int timePass = 0;

                        //ResetMediaPlayer();

                        //int idAudioTrain = -1;

                        ResultTime.Content = $"{timePass}/3";

                        ResultFinal.Background = new SolidColorBrush(Colors.WhiteSmoke);
                        ResultFinal.Foreground = new SolidColorBrush(Colors.Black);
                        ResultFinal.Content = "TESTING";

                        if (File.Exists(modelFilePath) && (!EnableReTraining || Path.Exists(datasetDir)))
                        {
                            bool resultSound = false;
                            for (int idxSound = 0; idxSound < 3; idxSound++)
                            {

                                try
                                {
                                    //idAudioTrain++;
                                    //await viewModel.StartRecordingSavingAsync(idAudioTrain);

                                    if (PredictSegment(modelFilePath, idMicrophone))
                                    {
                                        //AddMediaPlayer(idAudioTrain, OxyColors.Green);
                                        //MediaPlayerScrollViewer.ScrollToEnd();

                                        timePass++;
                                        ResultTime.Content = $"{timePass}/3";

                                        resultSound = true;


                                        continue;

                                    }
                                    else
                                    {
                                        //AddMediaPlayer(idAudioTrain, OxyColors.Red);
                                        //MediaPlayerScrollViewer.ScrollToEnd();

                                        resultSound = false;


                                        break;
                                    }




                                }
                                catch (Exception ex)
                                {
                                    // Handle exceptions from recording or prediction
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

                            if (EnableReTraining && !Path.Exists(datasetDir))
                            {
                                MessageBox.Show("Dataset folder not found.");
                            }
                        }


                    });

                    BarcodeReady = false;




                    //ResultFinal.Background = new SolidColorBrush(Colors.WhiteSmoke);
                    //ResultFinal.Foreground = new SolidColorBrush(Colors.Black);
                    //ResultFinal.Content = "READY";

                    continue;
                }

                else
                {
                    continue;
                }
            }

        }

        private async void StartTest(object sender, RoutedEventArgs e)
        {
            ReadyTest = true;
            StartTestBtn.IsEnabled = false;
            StopTestBtn.IsEnabled = true;

            await Task.Run(() => StartTestHandler());

        }

        private async void StopTest(object sender, RoutedEventArgs e)
        {
            ReadyTest = false;

            StartTestBtn.IsEnabled = true;
            StopTestBtn.IsEnabled = false;
        }

        private void AddMediaPlayer(int indexMediaPlayer, OxyColor color)
        {
            string waveFile = Path.Combine(viewModel.BackendPath, $"audio\\{indexMediaPlayer}.wav");

            MediaPlayer mediaPlayer = new MediaPlayer();
            mediaPlayers.Add(mediaPlayer);

            mediaPlayer.Open(new Uri(waveFile));

            mediaPlayer.Stop();

            WrapPanel wrapPanel = new WrapPanel
            {
                HorizontalAlignment = System.Windows.HorizontalAlignment.Center,
                VerticalAlignment = System.Windows.VerticalAlignment.Center

            };

            System.Windows.Controls.Label lblStatus = new System.Windows.Controls.Label
            {
                Name = "lblStatus",
                Content = "Not playing...",
                HorizontalContentAlignment = System.Windows.HorizontalAlignment.Center,
            };

            Grid gridStatus = new Grid
            {
                VerticalAlignment = System.Windows.VerticalAlignment.Center,
                Margin = new Thickness(5, 0, 0, 0)
            };
            gridStatus.Children.Add(lblStatus);

            System.Windows.Controls.Button btnListen = new System.Windows.Controls.Button
            {
                Name = "btnListen",

                Content = new IconBlock
                {
                    Icon = IconChar.Play,
                },

                Margin = new Thickness(5, 0, 0, 0),
                Width = 30,
                Height = 30,
                BorderThickness = new Thickness(0),
                Background = new SolidColorBrush(Colors.White),
                Foreground = new SolidColorBrush(Colors.LawnGreen)
            };

            btnListen.Click += (sender, e) => btnListen_Click(sender, e, mediaPlayer);


            System.Windows.Controls.CheckBox chkEnable_NG = new System.Windows.Controls.CheckBox
            {
                Name = $"chkEnable_NG",
                Content = "NG",
                IsChecked = false,
                Foreground = new SolidColorBrush(Colors.Red)
            };

            chkEnable_NG.Checked += (sender, e) => chkEnable_Checked_NG(sender, e, mediaPlayer, indexMediaPlayer);
            chkEnable_NG.Unchecked += (sender, e) => chkEnable_Unchecked_NG(sender, e, mediaPlayer, indexMediaPlayer);

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

            chkEnable_OK.Checked += (sender, e) => chkEnable_Checked_OK(sender, e, mediaPlayer, indexMediaPlayer);
            chkEnable_OK.Unchecked += (sender, e) => chkEnable_Unchecked_OK(sender, e, mediaPlayer, indexMediaPlayer);

            Grid gridEnable_OK = new Grid
            {
                VerticalAlignment = System.Windows.VerticalAlignment.Center,
                Margin = new Thickness(5, 0, 0, 0)
            };
            gridEnable_OK.Children.Add(chkEnable_OK);


            PlotModel plotModel = new PlotModel { };
            plotModel.Series.Add(new LineSeries { LineStyle = LineStyle.Solid, Color = color });
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom });
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Minimum = -100, Maximum = 25 });



            OxyPlot.Wpf.PlotView plotView = new OxyPlot.Wpf.PlotView
            {
                Model = plotModel,
                Width = 500,
                Height = 100,
            };

            wrapPanel.Children.Add(plotView);
            wrapPanel.Children.Add(gridStatus);
            wrapPanel.Children.Add(btnListen);
            //wrapPanel.Children.Add(gridEnable_NG);
            //wrapPanel.Children.Add(gridEnable_OK);

            MediaPlayerStack.Children.Add(wrapPanel);

            DispatcherTimer timer = new DispatcherTimer();
            timer.Interval = TimeSpan.FromSeconds(1);
            timer.Tick += (sender, e) => timer_Tick(sender, e, mediaPlayer, lblStatus);
            timer.Start();

            PCBViewModel.UpdatePlot(plotModel, viewModel.AudioRecorderTestingPage.GetAudioData());


        }

        public bool PredictSegment(string modelPath, int idMicrophone)
        {
            bool result = false;
            StringBuilder buffer = new StringBuilder();

            try
            {
                // Create a NamedPipeClientStream
                using (var pipeClient = new NamedPipeClientStream(".", pipeName, PipeDirection.InOut, PipeOptions.None))
                {
                    // Connect to the pipe
                    pipeClient.Connect();

                    // Write to the pipe
                    string message = $"test@{modelPath}";
                    byte[] messageBytes = Encoding.UTF8.GetBytes(message);
                    pipeClient.Write(messageBytes, 0, messageBytes.Length);

                    // Read the response
                    byte[] byteBuffer = new byte[64 * 1024]; // 64 KB buffer

                    while (true)
                    {
                        int bytesRead = pipeClient.Read(byteBuffer, 0, byteBuffer.Length);
                        string receivedData = Encoding.UTF8.GetString(byteBuffer, 0, bytesRead);
                        buffer.Append(receivedData);

                        if (buffer.ToString().Contains("endingtest"))
                        {
                            break;
                        }
                        // Check if we have a complete message in the buffer
                        while (buffer.ToString().Contains("response:"))
                        {
                            // Find the start and end of a message
                            int startIndex = buffer.ToString().IndexOf("response:");
                            int endIndex = buffer.ToString().IndexOf(":end", startIndex); 

                            // If a complete message is found
                            if (endIndex != -1)
                            {
                                // Extract the message
                                string fullMessage = buffer.ToString().Substring(startIndex, endIndex - startIndex + 1);

                                // Process the extracted message
                                //string result = ProcessResponse(fullMessage);
                                try
                                {
                                    // Extract the part after "response:"
                                    string resultPart = fullMessage.Split(new[] { "response:" }, StringSplitOptions.None)[1].Trim();

                                    if (resultPart != null && resultPart.Contains("True")) {
                                        result = true;
                                    }

                                }
                                catch (Exception ex)
                                {
                                    Console.WriteLine("Error processing response: " + ex.Message);
                                }

                              
                                // Remove the processed message from the buffer
                                buffer.Remove(startIndex, endIndex + 4 - startIndex);
                             

                            }
                            else
                            {
                                // No complete message found yet, wait for more data
                                break;
                            }
                        }


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

            return result ;


        }

        public bool Predict(int index, string modelPath)
        {
            try
            {
                // Create a NamedPipeClientStream
                using (var pipeClient = new NamedPipeClientStream(".", pipeName, PipeDirection.InOut, PipeOptions.None))
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
                            string resultPart = response.Split(new[] { "response:" }, StringSplitOptions.None)[1].Trim();

                            return resultPart == "True";

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

            return false;
        }


        private async void AutoScaleButton_Click(object sender, RoutedEventArgs e)
        {
            await viewModel.AutoScaleButtonAsync();
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

                // Open the file with the default application
            }

        }


        private void CheckCommunication_Init()
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
    }
}