using System.IO;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Axes;
using NAudio.Wave;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using SoundKit.Audio;

namespace SoundKit
{
    public class PCBViewModel
    {
        public const double seconds = 0.5;
        public const double N = 4;

        public string BackendPath = string.Empty;
        public PlotModel PlotModelLearningPage { get; private set; }
        public PlotModel PlotModelRealtimeTime { get; private set; }
        public PlotModel PlotModelRealtimeFreq { get; private set; }


        public AudioRecorder AudioRecorderLearningPage;
        public AudioRecorder AudioRecorderTestingPage;

        private RealtimeAudioRecorder audioRecorderRealtimePage;


        public PlotModel PlotModelTime_Item1_OK { get; private set; }
        public PlotModel PlotModelFreq_Item1_OK { get; private set; }
        public PlotModel PlotModelTime_Item1_NG { get; private set; }
        public PlotModel PlotModelFreq_Item1_NG { get; private set; }

        public PlotModel PlotModelTime_Item2_OK { get; private set; }
        public PlotModel PlotModelFreq_Item2_OK { get; private set; }
        public PlotModel PlotModelTime_Item2_NG { get; private set; }
        public PlotModel PlotModelFreq_Item2_NG { get; private set; }

        public PlotModel PlotModelTime_Item3_OK { get; private set; }
        public PlotModel PlotModelFreq_Item3_OK { get; private set; }
        public PlotModel PlotModelTime_Item3_NG { get; private set; }
        public PlotModel PlotModelFreq_Item3_NG { get; private set; }


        public PCBViewModel(int idMic)
        {
            PlotModelLearningPage = new PlotModel { Title = "Audio Waveform" };
            PlotModelLearningPage.Series.Add(new LineSeries { LineStyle = LineStyle.Solid });
            PlotModelLearningPage.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = $"Time  ({(int)(1 / (float)AudioRecorder.SampleRate * 1000000)} microsec per sample) " });
            PlotModelLearningPage.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Amplitude (dB)", Minimum = -100, Maximum = 10 });

            AudioRecorderLearningPage = new AudioRecorder(idMic, seconds);
            AudioRecorderTestingPage = new AudioRecorder(idMic, seconds);

            // Initialize OxyPlot
            PlotModelRealtimeTime = new PlotModel { Title = "Time Domain" };
            PlotModelRealtimeTime.Series.Add(new LineSeries() { Color = OxyColor.FromRgb(0, 135, 210) });
            PlotModelRealtimeTime.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = $"Time ({(int)(1 / (float)AudioRecorder.SampleRate * 1000000)} microsec per sample) " });
            PlotModelRealtimeTime.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Amplitude (dB)", Minimum = -100, Maximum = 25 });


            // Initialize OxyPlot
            PlotModelRealtimeFreq = new PlotModel { Title = "Frequency Domain" };
            PlotModelRealtimeFreq.Series.Add(new LineSeries() { Color = OxyColor.FromRgb(0, 135, 210) });
            PlotModelRealtimeFreq.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Frequency (Hz)" });
            PlotModelRealtimeFreq.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Magnitude", Minimum = 0, Maximum = 5000 });

            audioRecorderRealtimePage = new RealtimeAudioRecorder(idMic, seconds);

            //

            PlotModelTime_Item1_OK = new PlotModel { Title = "Time Domain" };
            PlotModelTime_Item1_OK.Series.Add(new LineSeries() { Color = OxyColors.Green });
            PlotModelTime_Item1_OK.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = $"Time ({(int)(1 / (float)AudioRecorder.SampleRate * 1000000)} microsec per sample) " });
            PlotModelTime_Item1_OK.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Amplitude (dB)", Minimum = -100, Maximum = 25 });

            PlotModelFreq_Item1_OK = new PlotModel { Title = "Frequency Domain" };
            PlotModelFreq_Item1_OK.Series.Add(new LineSeries() { Color = OxyColors.Green });
            PlotModelFreq_Item1_OK.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Frequency (Hz)" });
            PlotModelFreq_Item1_OK.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Magnitude", Minimum = 0, Maximum = 5000 });


            PlotModelTime_Item1_NG = new PlotModel { Title = "Time Domain" };
            PlotModelTime_Item1_NG.Series.Add(new LineSeries() { Color = OxyColors.Red });
            PlotModelTime_Item1_NG.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = $"Time ({(int)(1 / (float)AudioRecorder.SampleRate * 1000000)} microsec per sample) " });
            PlotModelTime_Item1_NG.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Amplitude (dB)", Minimum = -100, Maximum = 25 });

            PlotModelFreq_Item1_NG = new PlotModel { Title = "Frequency Domain" };
            PlotModelFreq_Item1_NG.Series.Add(new LineSeries() { Color = OxyColors.Red });
            PlotModelFreq_Item1_NG.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Frequency (Hz)" });
            PlotModelFreq_Item1_NG.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Magnitude", Minimum = 0, Maximum = 5000 });

            //
            PlotModelTime_Item2_OK = new PlotModel { Title = "Time Domain" };
            PlotModelTime_Item2_OK.Series.Add(new LineSeries() { Color = OxyColors.Green });
            PlotModelTime_Item2_OK.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = $"Time ({(int)(1 / (float)AudioRecorder.SampleRate * 1000000)} microsec per sample) " });
            PlotModelTime_Item2_OK.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Amplitude (dB)", Minimum = -100, Maximum = 25 });

            PlotModelFreq_Item2_OK = new PlotModel { Title = "Frequency Domain" };
            PlotModelFreq_Item2_OK.Series.Add(new LineSeries() { Color = OxyColors.Green });
            PlotModelFreq_Item2_OK.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Frequency (Hz)" });
            PlotModelFreq_Item2_OK.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Magnitude", Minimum = 0, Maximum = 5000 });


            PlotModelTime_Item2_NG = new PlotModel { Title = "Time Domain" };
            PlotModelTime_Item2_NG.Series.Add(new LineSeries() { Color = OxyColors.Red });
            PlotModelTime_Item2_NG.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = $"Time ({(int)(1 / (float)AudioRecorder.SampleRate * 1000000)} microsec per sample) " });
            PlotModelTime_Item2_NG.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Amplitude (dB)", Minimum = -100, Maximum = 25 });

            PlotModelFreq_Item2_NG = new PlotModel { Title = "Frequency Domain" };
            PlotModelFreq_Item2_NG.Series.Add(new LineSeries() { Color = OxyColors.Red });
            PlotModelFreq_Item2_NG.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Frequency (Hz)" });
            PlotModelFreq_Item2_NG.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Magnitude", Minimum = 0, Maximum = 5000 });

            //
            PlotModelTime_Item3_OK = new PlotModel { Title = "Time Domain" };
            PlotModelTime_Item3_OK.Series.Add(new LineSeries() { Color = OxyColors.Green });
            PlotModelTime_Item3_OK.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = $"Time ({(int)(1 / (float)AudioRecorder.SampleRate * 1000000)} microsec per sample) " });
            PlotModelTime_Item3_OK.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Amplitude (dB)", Minimum = -100, Maximum = 25 });

            PlotModelFreq_Item3_OK = new PlotModel { Title = "Frequency Domain" };
            PlotModelFreq_Item3_OK.Series.Add(new LineSeries() { Color = OxyColors.Green });
            PlotModelFreq_Item3_OK.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Frequency (Hz)" });
            PlotModelFreq_Item3_OK.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Magnitude", Minimum = 0, Maximum = 5000 });


            PlotModelTime_Item3_NG = new PlotModel { Title = "Time Domain" };
            PlotModelTime_Item3_NG.Series.Add(new LineSeries() { Color = OxyColors.Red });
            PlotModelTime_Item3_NG.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = $"Time ({(int)(1 / (float)AudioRecorder.SampleRate * 1000000)} microsec per sample) " });
            PlotModelTime_Item3_NG.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Amplitude (dB)", Minimum = -100, Maximum = 25 });

            PlotModelFreq_Item3_NG = new PlotModel { Title = "Frequency Domain" };
            PlotModelFreq_Item3_NG.Series.Add(new LineSeries() { Color = OxyColors.Red });
            PlotModelFreq_Item3_NG.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Frequency (Hz)" });
            PlotModelFreq_Item3_NG.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Magnitude", Minimum = 0, Maximum = 5000 });
        }


        public void UpdatePlotRealtime(object sender, EventArgs e)
        {
            try
            {
                if (PlotModelRealtimeTime.Series[0] is LineSeries lineSeries)
                {
                    lineSeries.Points.Clear();
                    for (int i = 0; i < audioRecorderRealtimePage.AudioDataIndB.Count; i++)
                    {
                        lineSeries.Points.Add(new DataPoint(i, audioRecorderRealtimePage.AudioDataIndB[i]));
                    }

                    PlotModelRealtimeTime.InvalidatePlot(true);
                }

                // FFT
                var fftBuffer = new Complex32[audioRecorderRealtimePage.BufferSize];
                if (audioRecorderRealtimePage.BufferSize <= audioRecorderRealtimePage.AudioData.Count)
                {

                    for (int i = 0; i < audioRecorderRealtimePage.BufferSize; i++)
                    {
                        fftBuffer[i] = new Complex32(audioRecorderRealtimePage.AudioData[audioRecorderRealtimePage.AudioData.Count + i - audioRecorderRealtimePage.BufferSize], 0);
                    }

                    Fourier.Forward(fftBuffer);

                    // Frequency domain plot
                    if (PlotModelRealtimeFreq.Series[0] is LineSeries fftSeries)
                    {
                        fftSeries.Points.Clear();
                        for (int i = 0; i < fftBuffer.Length / 2; i++) // Only need half of the FFT results
                        {

                            double magnitude = Math.Sqrt(Math.Pow(fftBuffer[i].Real, 2) + Math.Pow(fftBuffer[i].Imaginary, 2));
                            double frequency = i * (AudioRecorder.SampleRate / (double)audioRecorderRealtimePage.BufferSize);


                            fftSeries.Points.Add(new DataPoint(frequency, magnitude));

                        }

                        PlotModelRealtimeFreq.InvalidatePlot(true);
                    }
                }
            }
            catch
            {

            }



        }

        public void CaptureData(string plotModelKey)
        {// Dictionary to map strings to PlotModel instances

            Task.Delay((int)(1000 * seconds));

            Dictionary<string, PlotModel> plotModelsTime = new()
            {
                { "PlotModel_Item1_OK", PlotModelTime_Item1_OK },  // Add your actual PlotModels here
                { "PlotModel_Item1_NG", PlotModelTime_Item1_NG },
                { "PlotModel_Item2_OK", PlotModelTime_Item2_OK },  // Add your actual PlotModels here
                { "PlotModel_Item2_NG", PlotModelTime_Item2_NG },
                { "PlotModel_Item3_OK", PlotModelTime_Item3_OK },  // Add your actual PlotModels here
                { "PlotModel_Item3_NG", PlotModelTime_Item3_NG }
            };


            if (plotModelsTime.ContainsKey(plotModelKey))
            {
                PlotModel plotModel = plotModelsTime[plotModelKey];

                if (plotModel.Series[0] is LineSeries lineSeries)
                {
                    lineSeries.Points.Clear();

                    var lastSamples = audioRecorderRealtimePage.AudioDataIndB
                        .Skip(Math.Max(0, audioRecorderRealtimePage.AudioDataIndB.Count - audioRecorderRealtimePage.BufferSize))
                        .Take(audioRecorderRealtimePage.BufferSize);

                    int i = 0;
                    foreach (var sample in lastSamples)
                    {
                        lineSeries.Points.Add(new DataPoint(i, sample));
                        i++;
                    }

                    plotModel.InvalidatePlot(true);
                }
            }

            Dictionary<string, PlotModel> plotModelsFreq = new()
            {
                { "PlotModel_Item1_OK", PlotModelFreq_Item1_OK },  // Add your actual PlotModels here
                { "PlotModel_Item1_NG", PlotModelFreq_Item1_NG },
                { "PlotModel_Item2_OK", PlotModelFreq_Item2_OK },  // Add your actual PlotModels here
                { "PlotModel_Item2_NG", PlotModelFreq_Item2_NG },
                { "PlotModel_Item3_OK", PlotModelFreq_Item3_OK },  // Add your actual PlotModels here
                { "PlotModel_Item3_NG", PlotModelFreq_Item3_NG }
            };

            if (plotModelsFreq.ContainsKey(plotModelKey))
            {
                PlotModel plotModel = plotModelsFreq[plotModelKey];
                // FFT
                var fftBuffer = new Complex32[audioRecorderRealtimePage.BufferSize];
                if (audioRecorderRealtimePage.BufferSize <= audioRecorderRealtimePage.AudioData.Count)
                {

                    for (int i = 0; i < audioRecorderRealtimePage.BufferSize; i++)
                    {
                        fftBuffer[i] = new Complex32(audioRecorderRealtimePage.AudioData[audioRecorderRealtimePage.AudioData.Count + i - audioRecorderRealtimePage.BufferSize], 0);
                    }

                    Fourier.Forward(fftBuffer);

                    // Frequency domain plot
                    if (plotModel.Series[0] is LineSeries fftSeries)
                    {
                        fftSeries.Points.Clear();
                        for (int i = 0; i < fftBuffer.Length / 2; i++) // Only need half of the FFT results
                        {

                            double magnitude = Math.Sqrt(Math.Pow(fftBuffer[i].Real, 2) + Math.Pow(fftBuffer[i].Imaginary, 2));
                            double frequency = i * (AudioRecorder.SampleRate / (double)audioRecorderRealtimePage.BufferSize);


                            fftSeries.Points.Add(new DataPoint(frequency, magnitude));

                        }

                        plotModel.InvalidatePlot(true);
                    }
                }
            }
        }


        public static void UpdatePlot(PlotModel plotModel, double[] audioData)
        {
            var lineSeries = (LineSeries)plotModel.Series[0];
            lineSeries.Points.Clear();
            for (int i = 0; i < audioData.Length; i++)
            {
                lineSeries.Points.Add(new DataPoint(i, audioData[i]));
            }
            plotModel.InvalidatePlot(true);
        }

        public async Task AutoScaleButtonAsync()
        {
            PlotModelRealtimeTime.ResetAllAxes();
            PlotModelRealtimeTime.InvalidatePlot(true);
            PlotModelRealtimeFreq.ResetAllAxes();
            PlotModelRealtimeFreq.InvalidatePlot(true);
        }

        public async Task StartRecordingAsync(int duration)
        {
            AudioRecorderLearningPage.StartRecording();
            AudioRecorderLearningPage.RecordingDuration = TimeSpan.FromSeconds(duration);
            AudioRecorderLearningPage.RecordingStartTime = DateTime.Now;
            await Task.Delay(duration * 1000);
            AudioRecorderLearningPage.StopRecording();
            UpdatePlot(PlotModelLearningPage, AudioRecorderLearningPage.GetAudioData());
        }
        public async Task StopRecordingAsync()
        {
            AudioRecorderLearningPage.StopRecording();
            UpdatePlot(PlotModelLearningPage, AudioRecorderLearningPage.GetAudioData());
        }

        public async Task StartRecordingSavingAsync(int index)
        {
            AudioRecorderTestingPage.StartRecording();
            AudioRecorderTestingPage.RecordingDuration = TimeSpan.FromSeconds(N*seconds);
            AudioRecorderTestingPage.RecordingStartTime = DateTime.Now;

            await Task.Delay((int)(1000 * N * seconds));
            AudioRecorderTestingPage.StopRecording();
            string savePath = Path.Combine(BackendPath, $"audio\\{index}.wav");
            AudioRecorderTestingPage.SaveAudioData(savePath);
        }
    }
}
