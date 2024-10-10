using NAudio.Wave;
using System.Windows.Threading;


namespace SoundKit.Audio
{
    public class RealtimeAudioRecorder : AudioRecorder
    {
        private DispatcherTimer timer;

        public int BufferSize;

        public RealtimeAudioRecorder(int idMic, double pSeconds) : base(idMic, pSeconds)
        {
            seconds = pSeconds;
            BufferSize = (int)(SampleRate * seconds);

            waveIn = new WaveInEvent
            {
                WaveFormat = new WaveFormat(SampleRate, BitsPerSample, Channels)
            };

            waveIn.DeviceNumber = idMic;

            waveIn.DataAvailable += OnDataAvailable;
            waveIn.StartRecording();
        }



    }


}
