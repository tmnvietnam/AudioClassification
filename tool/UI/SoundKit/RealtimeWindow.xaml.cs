
using System.Windows;
using System.Windows.Threading;

namespace SoundKit
{
    /// <summary>
    /// Interaction logic for Training.xaml
    /// </summary>
    public partial class RealtimeWindow : Window
    {
        private DispatcherTimer timer;

        private readonly RealtimeViewModel viewModel;
        public RealtimeWindow(string fixtureName, int idMicrophone)
        {
            InitializeComponent();
            viewModel = new RealtimeViewModel(idMicrophone);

            this.Title = $"{fixtureName} - Realtime";
            DataContext = viewModel;

            // Set up a timer to update the plot periodically
            timer = new DispatcherTimer
            {
                Interval = TimeSpan.FromMilliseconds(50)
            };
            timer.Tick += viewModel.UpdatePlotRealtime;
            timer.Start();
        }      

        private async void AutoScaleButton_Click(object sender, RoutedEventArgs e)
        {
            await viewModel.AutoScaleButtonAsync();
        }

    }
}
