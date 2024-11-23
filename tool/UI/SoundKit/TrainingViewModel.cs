using OxyPlot.Axes;
using OxyPlot.Series;
using OxyPlot;
using SoundKit.Audio;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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
    internal class TrainingViewModel
    {
        public double windowLength = 1;

        public PlotModel PlotModel { get; private set; }

        public AudioRecorder AudioRecorder;

        public TrainingViewModel()
        {
            PlotModel = new PlotModel { Title = "Audio Waveform" };
            PlotModel.Series.Add(new LineSeries { LineStyle = LineStyle.Solid });
            PlotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = $"Time  ({(int)(1 / (float)AudioRecorder.SampleRate * 1000000)} microsec per sample) " });
            PlotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Amplitude", Minimum = -32767, Maximum = 32767 });
            AudioRecorder = new();
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


        public async Task StartRecordingAsync(int duration)
        {
            AudioRecorder.StartRecording();
            AudioRecorder.RecordingDuration = TimeSpan.FromSeconds(duration);
            AudioRecorder.RecordingStartTime = DateTime.Now;
            await Task.Delay(duration * 1000);
            AudioRecorder.StopRecording();
            UpdatePlot(PlotModel, AudioRecorder.GetAudioData());
        }
    }
}
