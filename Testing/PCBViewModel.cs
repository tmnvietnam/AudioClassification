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
        public const double seconds = 1;
        public const double N = 4;

        public string BackendPath = string.Empty;
        public PlotModel PlotModelRealtimeTime { get; private set; }
        public PlotModel PlotModelRealtimeFreq { get; private set; }


        public AudioRecorder AudioRecorderTestingPage;

        private RealtimeAudioRecorder audioRecorderRealtimePage;


        public PCBViewModel(int idMic)
        {

            AudioRecorderTestingPage = new AudioRecorder(idMic, seconds);

            // Initialize OxyPlot
            PlotModelRealtimeTime = new PlotModel { Title = "Time Domain" };
            PlotModelRealtimeTime.Series.Add(new LineSeries() { Color = OxyColor.FromRgb(0, 135, 210) });
            PlotModelRealtimeTime.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = $"Time ({(int)(1 / (float)AudioRecorder.SampleRate * 1000000)} microsec per sample) " });
            PlotModelRealtimeTime.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Amplitude", Minimum = -25000, Maximum = 25000 });


            // Initialize OxyPlot
            PlotModelRealtimeFreq = new PlotModel { Title = "Frequency Domain" };
            PlotModelRealtimeFreq.Series.Add(new LineSeries() { Color = OxyColor.FromRgb(0, 135, 210) });
            PlotModelRealtimeFreq.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Frequency (Hz)" , Minimum = 5000, Maximum = 9500 });
            PlotModelRealtimeFreq.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Magnitude", Minimum = 0, Maximum = 500 });

            audioRecorderRealtimePage = new RealtimeAudioRecorder(idMic, seconds);

            
        }


        public void UpdatePlotRealtime(object sender, EventArgs e)
        {
            try
            {
                if (PlotModelRealtimeTime.Series[0] is LineSeries lineSeries)
                {
                    lineSeries.Points.Clear();
                    for (int i = 0; i < audioRecorderRealtimePage.AudioData.Count; i++)
                    {
                        lineSeries.Points.Add(new DataPoint(i, audioRecorderRealtimePage.AudioData[i]));
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

        public static void UpdatePlot(PlotModel plotModel, short[] audioData)
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
