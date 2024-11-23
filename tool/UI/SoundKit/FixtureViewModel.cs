using OxyPlot;
using OxyPlot.Series;

namespace SoundKit
{
    public class FixtureViewModel
    {
        public FixtureViewModel()
        {
            
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
    }
}
