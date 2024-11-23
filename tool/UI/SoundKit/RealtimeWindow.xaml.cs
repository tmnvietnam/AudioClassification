
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
        public RealtimeWindow(int idMicrophone)
        {
            InitializeComponent();
            viewModel = new RealtimeViewModel(idMicrophone);

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
