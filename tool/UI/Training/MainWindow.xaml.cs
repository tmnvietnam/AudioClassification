﻿using NAudio.Wave;
using Training.Audio;
using System.Diagnostics;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using System.IO.Pipes;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Forms;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using MessageBox = System.Windows.MessageBox;
using Button = System.Windows.Controls.Button;
using Path = System.IO.Path;
using System.Windows.Shapes;
using MathNet.Numerics;
using Python.Runtime;


namespace Training
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {

        public const string ServicePath = "C:\\Users\\ADMIN\\Documents\\main\\working\\Audio.Classification\\tool\\service";

        private Point startPoint;

        private Point endPoint;

        private int startSample;

        private int endSample;
        private DispatcherTimer timer;

        private readonly MainViewModel viewModel;

        private double minX = 85;
        private double maxX = 1363;

        private string datasetDir = string.Empty;
        private string ngFolderPath = string.Empty;
        private string okFolderPath = string.Empty;
        private string modelFilePath = string.Empty;

        private int epochs = 0;
        private int batchSize = 0;
        private int patience = 0;

        static string[] labels = new string[] { "NG", "OK", "NG.PCB" };

        private dynamic aiCore;



        private string pipeName = string.Empty;

        public MainWindow()
        {
            InitializeComponent();
            InitPython();

            LoadMicrophones();

            this.Closed += Window_Closed;

            viewModel = new();
            DataContext = viewModel;

        }


        void InitPython()
        {
            string pathToVirtualEnv = @"C:\Users\ADMIN\AppData\Local\Programs\Python\Python39";

            var path = Environment.GetEnvironmentVariable("PATH").TrimEnd(';');
            path = string.IsNullOrEmpty(path) ? pathToVirtualEnv : path + ";" + pathToVirtualEnv;
            Environment.SetEnvironmentVariable("PATH", path, EnvironmentVariableTarget.Process);
            Environment.SetEnvironmentVariable("PATH", pathToVirtualEnv, EnvironmentVariableTarget.Process);
            Environment.SetEnvironmentVariable("PYTHONHOME", pathToVirtualEnv, EnvironmentVariableTarget.Process);
            Environment.SetEnvironmentVariable("PYTHONPATH", $@"{pathToVirtualEnv}\\Lib\\site-packages;{pathToVirtualEnv}\\Lib;{ServicePath}", EnvironmentVariableTarget.Process);

            PythonEngine.PythonHome = pathToVirtualEnv;
            PythonEngine.PythonPath = PythonEngine.PythonPath + ";" + Environment.GetEnvironmentVariable("PYTHONPATH", EnvironmentVariableTarget.Process);
            PythonEngine.Initialize();
            PythonEngine.BeginAllowThreads();


            using (Py.GIL())
            {
                aiCore = Py.Import("aicore");
            }
        }




        private void UpdateProgressBar(double progress)
        {
            Dispatcher.Invoke(() => RecordingProgressBar.Value = progress);
        }

        private async void StartRecording_Click(object sender, RoutedEventArgs e)
        {
            // Start recording and update the plot
            await viewModel.StartRecordingAsync(int.Parse(TextBoxDuration.Text));
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

        private void SaveRecording_Click(object sender, RoutedEventArgs e)
        {
            string folderPath = string.Empty;
            // Cast sender to a Button to access properties
            Button clickedButton = sender as Button;

            if (clickedButton != null)
            {
                // Check the Content or Background property to distinguish between buttons
                if ((clickedButton.Content as StackPanel)?.Children.OfType<TextBlock>().FirstOrDefault()?.Text == "OK")
                {
                    // Handle "OK" button click
                    folderPath = Path.Combine(datasetDir, $"OK");

                }
                else if ((clickedButton.Content as StackPanel)?.Children.OfType<TextBlock>().FirstOrDefault()?.Text == "NG")
                {
                    // Handle "NG" button click
                    folderPath = Path.Combine(datasetDir, $"NG");

                }
            }

            if (!File.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }

            if (Path.Exists(folderPath))
            {
                string fileName = GetNextFileName(folderPath);
                string filePath = Path.Combine(folderPath, fileName);

                viewModel.AudioRecorder.ExtractAndSaveAudioSegment(filePath, startSample, endSample);

                if ((clickedButton.Content as StackPanel)?.Children.OfType<TextBlock>().FirstOrDefault()?.Text == "OK")
                {
                    NumberDataSetOK.Content = $"{CountFilesInDirectory(Path.Combine(datasetDir, $"OK"))}";

                }
                else if ((clickedButton.Content as StackPanel)?.Children.OfType<TextBlock>().FirstOrDefault()?.Text == "NG")
                {
                    NumberDataSetNG.Content = $"{CountFilesInDirectory(Path.Combine(datasetDir, $"NG"))}";
                }


            }
            else
            {
                MessageBox.Show("Dataset folder not found.");
            }
        }


        private void Window_Closed(object sender, EventArgs e)
        {
            PythonEngine.Shutdown();
            PythonEngine.EndAllowThreads(0);

            viewModel.AudioRecorder.Dispose();

        }

        private void Rectangle_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (sender is Rectangle rectangle)
            {
                rectangle.CaptureMouse();
            }
        }


        private void Rectangle_MouseMove(object sender, System.Windows.Input.MouseEventArgs e)
        {
            if (sender is Rectangle rectangle && rectangle.IsMouseCaptured)
            {
                if (viewModel.AudioRecorder.GetAudioData().Length > 0)
                {
                    double tempStartPointX = e.GetPosition(OverlayCanvas).X;
                    double tempEndPointX = (int)(tempStartPointX + (double)(AudioRecorder.SampleRate * viewModel.windowLength) / viewModel.AudioRecorder.GetAudioData().Length * (maxX - minX));

                    if (tempEndPointX <= maxX && tempStartPointX >= minX)
                    {
                        startPoint.X = tempStartPointX;
                        endPoint.X = tempEndPointX;

                        Canvas.SetLeft(SelectionCanvas, startPoint.X);
                        SelectionCanvas.Width = endPoint.X - startPoint.X;

                        startSample = (int)((startPoint.X - 85 >= 0 ? startPoint.X - 85 : 0) / (maxX - minX) * viewModel.AudioRecorder.GetAudioData().Length);
                        endSample = Math.Min(startSample + (int)(AudioRecorder.SampleRate * viewModel.windowLength) - 1, viewModel.AudioRecorder.GetAudioData().Length - 1);
                    }
                }
            }
        }

        private void Rectangle_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (sender is Rectangle rectangle)
            {
                rectangle.ReleaseMouseCapture();  // Release mouse capture when done
            }
        }

        private void DatasetBrowseButton_Click(object sender, RoutedEventArgs e)
        {
            using (var folderDialog = new FolderBrowserDialog())
            {
                folderDialog.SelectedPath = @"C:\";

                if (folderDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    string selectedPath = folderDialog.SelectedPath;

                    DatasetTextBox.Text = selectedPath;
                    datasetDir = DatasetTextBox.Text;

                    NumberDataSetOK.Content = $"{CountFilesInDirectory(Path.Combine(datasetDir, $"OK"))}";
                    NumberDataSetNG.Content = $"{CountFilesInDirectory(Path.Combine(datasetDir, $"NG"))}";

                }
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
        public int CountFilesInDirectory(string folderPath)
        {
            int fileCount = 0;

            try
            {
                // Get all files in the current directory
                fileCount += Directory.GetFiles(folderPath).Length;

            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred: {ex.Message}");
            }

            return fileCount;
        }

        private void SaveModelFileButton_Click(object sender, RoutedEventArgs e)
        {
            // Create an instance of SaveFileDialog
            SaveFileDialog saveFileDialog = new SaveFileDialog
            {
                FileName = "model", // Default file name
                DefaultExt = ".keras" // Default file extension
            };

            bool result = saveFileDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK;
            // Process the result
            if (result == true)
            {
                string filePath = saveFileDialog.FileName;

                File.Copy(Path.Combine(ServicePath, "model.keras"), filePath, overwrite: true);
                MessageBox.Show("Model file saved successfully!", "Success", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }



        private void RunTraining(string datasetPath, int epochs, int batchSize, int patience)
        {
            using (Py.GIL()) // Acquire the Global Interpreter Lock
            {
                try
                {
                    Dispatcher.Invoke(() => TrainingResult.Text = "Training...");

                    // Call the Python train function
                    dynamic res = aiCore.train(datasetPath, epochs, batchSize, patience);

                    Dispatcher.Invoke(() => TrainingResult.Text = $"Accuracy: {res[0]*100:F2}%   Loss: {res[1]*100:F2}%");

                }
                catch (PythonException ex)
                {
                    Debug.WriteLine($"Python error: {ex.Message}");
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"C# error: {ex.Message}");

                }
            }
        }


        //public async Task TrainAsync(string datasetPath)
        //{
        //    try
        //    {
        //        TrainingResult.Text = $"Processing...";
        //        // Create a NamedPipeClientStream
        //        using (var pipeClient = new NamedPipeClientStream(".", pipeName, PipeDirection.InOut, PipeOptions.Asynchronous))
        //        {
        //            // Connect to the pipe
        //            await pipeClient.ConnectAsync();

        //            epochs = int.Parse(EpochTxt.Text);
        //            batchSize = int.Parse(BatchTxt.Text);

        //            // Write to the pipe
        //            string message = $"train@{datasetPath}@{epochs}@{batchSize}";
        //            byte[] messageBytes = Encoding.UTF8.GetBytes(message);
        //            await pipeClient.WriteAsync(messageBytes, 0, messageBytes.Length);

        //            // Read the response
        //            byte[] buffer = new byte[64 * 1024]; // 64 KB buffer
        //            int bytesRead = await pipeClient.ReadAsync(buffer, 0, buffer.Length);
        //            string response = Encoding.UTF8.GetString(buffer, 0, bytesRead);


        //            // Parse and format the accuracy value
        //            if (response.Contains("response:"))
        //            {
        //                try
        //                {
        //                    // Extract the accuracy value from the response
        //                    string accuracyPart = response.Split(new[] { ":" }, StringSplitOptions.None)[1].Trim();
        //                    string lossPart = response.Split(new[] { ":" }, StringSplitOptions.None)[2].Trim();

        //                    // Parse the accuracy value to float
        //                    if (float.TryParse(accuracyPart, out float accuracy) && float.TryParse(lossPart, out float loss))
        //                    {
        //                        // Format the accuracy value to two decimal places
        //                        string formattedAccuracy = (accuracy * 100).ToString("F3");
        //                        string formattedLoss = (loss * 100).ToString("F3");


        //                        // Update the UI (assuming this is a WPF application)
        //                        TrainingResult.Text = $"Accuracy: {formattedAccuracy}%   Loss: {formattedLoss}%";
        //                        byte[] imageData = File.ReadAllBytes(Path.Combine(viewModel.BackendPath, "history.png"));

        //                        BitmapImage bitmapImage = new BitmapImage();
        //                        using (var stream = new MemoryStream(imageData))
        //                        {
        //                            bitmapImage.BeginInit();
        //                            bitmapImage.CacheOption = BitmapCacheOption.OnLoad; // To fully load the image before closing the stream
        //                            bitmapImage.StreamSource = stream;
        //                            bitmapImage.EndInit();
        //                        }

        //                        // Set the image source to the Image control in WPF
        //                        HistoryImg.Source = bitmapImage;

        //                    }
        //                    else
        //                    {
        //                        Console.WriteLine("Error: Unable to parse the accuracy value.");
        //                        TrainingResult.Text = "Error parsing";
        //                    }
        //                }
        //                catch (Exception ex)
        //                {
        //                    Console.WriteLine($"Parsing error: {ex.Message}");
        //                    TrainingResult.Text = "Error parsing";
        //                }
        //            }
        //            else
        //            {
        //                Console.WriteLine("Response does not contain 'accuracy:'");
        //                TrainingResult.Text = "No data";
        //            }

        //            // Update UI (assuming this is a WPF application)
        //            // Ensure you're on the UI thread if updating UI elements

        //            Console.WriteLine($"Received from server: {response}");
        //        }
        //    }
        //    catch (IOException ex)
        //    {
        //        Console.WriteLine($"I/O error: {ex.Message}");
        //    }
        //    catch (Exception ex)
        //    {
        //        Console.WriteLine($"Unexpected error: {ex.Message}");
        //    }
        //}

        private void TrainButton_Click(object sender, RoutedEventArgs e)
        {
            datasetDir = datasetDir.Replace("\\", "/");

            if (Path.Exists(datasetDir))
            {
                try
                {
                    epochs = int.Parse(EpochTxt.Text);
                    batchSize = int.Parse(BatchTxt.Text);
                    patience = (int)(epochs * 0.75);

                    Task.Run(() => RunTraining(datasetDir, epochs, batchSize, patience));

                }
                catch (PythonException ex)
                {
                    Debug.WriteLine($"Python error: {ex.Message}");
                }

            }
            else
            {
                MessageBox.Show("Dataset folder not found.");
            }

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
            MicrophoneComboBox.ItemsSource = microphones;


            // Set the first item (default microphone) as selected if there are any devices
            if (microphones.Count > 0)
            {
                MicrophoneComboBox.SelectedIndex = 0; // Automatically select the first device (default microphone)        
            }


        }

        private void ApplyClick(object sender, RoutedEventArgs e)
        {
            int selectedMicrophoneIndex = MicrophoneComboBox.SelectedIndex;

            if (viewModel != null)
            {
                viewModel.AudioRecorder = new AudioRecorder(selectedMicrophoneIndex);
                viewModel.AudioRecorder.ProgressChanged += UpdateProgressBar;
            }
        }


    }
}