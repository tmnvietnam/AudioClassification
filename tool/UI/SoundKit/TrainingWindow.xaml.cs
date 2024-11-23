using NAudio.Wave;
using Python.Runtime;
using SoundKit.Audio;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Forms;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.Windows.Threading;
using Button = System.Windows.Controls.Button;
using MessageBox = System.Windows.MessageBox;
using Path = System.IO.Path;

namespace SoundKit
{
    /// <summary>
    /// Interaction logic for Training.xaml
    /// </summary>
    public partial class TrainingWindow : Window
    {

        public string ServicePath = Properties.Settings.Default.ServicePath;

        private Point startPoint;

        private Point endPoint;

        private int startSample;

        private int endSample;
        private DispatcherTimer timer;

        private readonly TrainingViewModel viewModel;

        private double minX = 85;
        private double maxX = 1363;

        private string datasetDir = string.Empty;
        private string ngFolderPath = string.Empty;
        private string okFolderPath = string.Empty;
        private string modelFilePath = string.Empty;

        private int epochs = 0;
        private int batchSize = 0;
        private int patience = 0;

        private dynamic aiCore;


        private string pipeName = string.Empty;

        public TrainingWindow(int idMicrophone, string datasetDir)
        {
            InitializeComponent();

            this.datasetDir = datasetDir;
            this.Closed += WindowClosed;

            viewModel = new();
            viewModel.AudioRecorder = new AudioRecorder(idMicrophone);
            viewModel.AudioRecorder.ProgressChanged += UpdateProgressBar;

            DataContext = viewModel;
            InitPython();

        }

        void InitPython()
        {

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
            EnsureDirectoryExists(Path.Combine(datasetDir, $"OK"));
            EnsureDirectoryExists(Path.Combine(datasetDir, $"NG"));
            EnsureDirectoryExists(Path.Combine(datasetDir, $"NG.PCB"));
            EnsureDirectoryExists(Path.Combine(datasetDir, $"NG.BG"));

            NumberDataSetOK.Content = $"{CountFilesInDirectory(Path.Combine(datasetDir, $"OK"))}";
            NumberDataSetNG.Content = $"{CountFilesInDirectory(Path.Combine(datasetDir, $"NG"))}";

            NumberDataSetNG_BG.Content = $"{CountFilesInDirectory(Path.Combine(datasetDir, $"NG.BG"))}";
            NumberDataSetNG_PCB.Content = $"{CountFilesInDirectory(Path.Combine(datasetDir, $"NG.PCB"))}";

            string folderPath = string.Empty;
            string label = string.Empty;
            // Cast sender to a Button to access properties
            Button clickedButton = sender as Button;

            if (clickedButton != null)
            {
                label = (clickedButton.Content as StackPanel)?.Children.OfType<TextBlock>().FirstOrDefault()?.Text;
                folderPath = Path.Combine(datasetDir, label);

                string fileName = GetNextFileName(folderPath);
                string filePath = Path.Combine(folderPath, fileName);
                viewModel.AudioRecorder.ExtractAndSaveAudioSegment(filePath, startSample, endSample);
            }          

          
        }


        private void WindowClosed(object sender, EventArgs e)
        {
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

                    Dispatcher.Invoke(() => TrainingResult.Text = $"Accuracy: {(res[0] * 100):F2}%   Loss: {(res[1] * 100):F2}%");

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

       
    }
}
