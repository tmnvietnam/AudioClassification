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
    public class FixtureViewModel
    {
        public const double seconds = 0.15;

        public PlotModel PlotModelRealtimeTime { get; private set; }
        public PlotModel PlotModelRealtimeFreq { get; private set; }



        private RealtimeAudioRecorder audioRecorderRealtimePage;


        public FixtureViewModel(int idMic)
        {

            // Initialize OxyPlot
            PlotModelRealtimeTime = new PlotModel { Title = "Time Domain" };
            PlotModelRealtimeTime.Series.Add(new LineSeries() { Color = OxyColor.FromRgb(0, 135, 210) });
            PlotModelRealtimeTime.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = $"Time ({(int)(1 / (float)AudioRecorder.SampleRate * 1000000)} microsec per sample) " });
            PlotModelRealtimeTime.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Amplitude", Minimum = -25000, Maximum = 25000 });


            // Initialize OxyPlot
            PlotModelRealtimeFreq = new PlotModel { Title = "Frequency Domain" };
            PlotModelRealtimeFreq.Series.Add(new LineSeries() { Color = OxyColor.FromRgb(0, 135, 210) });
            PlotModelRealtimeFreq.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Frequency (Hz)" , Minimum = 0, Maximum = 11025 });
            PlotModelRealtimeFreq.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Magnitude", Minimum = 0, Maximum = 1500 });

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
                int windowSize = audioRecorderRealtimePage.BufferSize;
                float sigma = windowSize / 6f; // Standard deviation, you can tweak this value
                float[] gaussianWindow = new float[windowSize];

                // Calculate the Gaussian window values
                for (int i = 0; i < windowSize; i++)
                {
                    float n = i - (windowSize - 1) / 2f; // Center the window
                    gaussianWindow[i] = (float)Math.Exp(-0.5 * (n * n) / (sigma * sigma));
                }

                var fftBuffer = new Complex32[windowSize];
                // Copy and apply the Gaussian window to the audio data
                if (windowSize <= audioRecorderRealtimePage.AudioData.Count)
                {
                    for (int i = 0; i < windowSize; i++)
                    {
                        // Apply the Gaussian window to the audio data
                        float windowedSample = audioRecorderRealtimePage.AudioData[audioRecorderRealtimePage.AudioData.Count + i - windowSize] * gaussianWindow[i];
                        fftBuffer[i] = new Complex32(windowedSample, 0);
                    }

                    // Perform FFT
                    Fourier.Forward(fftBuffer);

                    // Frequency domain plot
                    if (PlotModelRealtimeFreq.Series[0] is LineSeries fftSeries)
                    {
                        fftSeries.Points.Clear();
                        for (int i = 0; i < fftBuffer.Length / 2; i++) // Only need half of the FFT results
                        {
                            double magnitude = Math.Sqrt(Math.Pow(fftBuffer[i].Real, 2) + Math.Pow(fftBuffer[i].Imaginary, 2));
                            double frequency = i * (AudioRecorder.SampleRate / (double)windowSize);

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

 
    }
}
