using NAudio.Wave;

namespace Training.Audio
{
    public class AudioRecorder : IDisposable
    {
        protected WaveInEvent waveIn;

        protected double windowLength = 1;

        private System.Timers.Timer progressTimer;

        public TimeSpan RecordingDuration;
        public DateTime RecordingStartTime;

        public List<short> AudioData = new List<short>();

        public event Action<double> ProgressChanged;

        // Default audio format parameters
        public const int SampleRate = 22050; // Sample rate in Hz
        public const int BitsPerSample = 16; // Bit depth
        public const int Channels = 1; // Mono

        public AudioRecorder(int idMic = 0)
        {

            waveIn = new WaveInEvent
            {
                WaveFormat = new WaveFormat(SampleRate, BitsPerSample, Channels)
            };

            waveIn.DeviceNumber = idMic;

            waveIn.DataAvailable += OnDataAvailable;
        }

        public void StartRecording()
        {
            if (waveIn == null) throw new InvalidOperationException("AudioRecorder is disposed.");

            AudioData.Clear();

            //AudioDataIndB.Clear();

            waveIn.StartRecording();
            StartProgressTimer();
        }

        public void StopRecording()
        {
            if (waveIn == null) throw new InvalidOperationException("AudioRecorder is disposed.");

            waveIn.StopRecording();
            StopProgressTimer();
        }

        protected void OnDataAvailable(object sender, WaveInEventArgs e)
        {
            try
            {
                int sampleCount = e.BytesRecorded / sizeof(short);
                var samples = new short[sampleCount];

                Buffer.BlockCopy(e.Buffer, 0, samples, 0, e.BytesRecorded);
                //var dBValues = ConvertToDB(samples);


                lock (AudioData)
                {
                    AudioData.AddRange(samples);
                }
                if (AudioData.Count > SampleRate * 10) // Limit the number of samples
                {
                    AudioData.RemoveRange(0, AudioData.Count - SampleRate * 10);
                }

            }
            catch
            {

            }

        }

        public short[] GetAudioData()
        {
            lock (AudioData)
            {
                return AudioData.ToArray();
            }
        }

        private void StartProgressTimer()
        {
            progressTimer = new System.Timers.Timer(50); // Check every 100ms
            progressTimer.Elapsed += (s, e) =>
            {
                var elapsed = DateTime.Now - RecordingStartTime;
                var progress = Math.Min(1.0, elapsed.TotalSeconds / RecordingDuration.TotalSeconds);
                ProgressChanged?.Invoke(progress * 100);
            };
            progressTimer.Start();
        }

        private void StopProgressTimer()
        {
            if (progressTimer != null)
            {
                progressTimer.Stop();
                progressTimer.Dispose();
                progressTimer = null;
            }
        }

        public void ExtractAndSaveAudioSegment(string outputFilePath, int startSample, int endSample)
        {
            if (startSample < 0 || endSample > AudioData.Count || startSample >= endSample)
            {
                throw new ArgumentOutOfRangeException("Invalid sample range specified.");
            }

            var extractedData = AudioData.Skip(startSample).Take(endSample - startSample).ToList();

            SaveAudioData(outputFilePath, extractedData);
        }

        public void SaveAudioData(string outputFilePath, List<short> audioData)
        {
            var waveFormat = new WaveFormat(SampleRate, BitsPerSample, 1);
            using (var writer = new WaveFileWriter(outputFilePath, waveFormat))
            {
                // Convert List<short> to byte array
                var byteArray = new byte[audioData.Count * sizeof(short)];
                Buffer.BlockCopy(audioData.ToArray(), 0, byteArray, 0, byteArray.Length);

                // Write the byte array to the file
                writer.Write(byteArray, 0, byteArray.Length);
            }
        }
        public void SaveAudioData(string outputFilePath)
        {
            SaveAudioData(outputFilePath, AudioData);
        }

        public void Dispose()
        {
            if (waveIn != null)
            {
                waveIn.Dispose();
                waveIn = null;
            }
            GC.SuppressFinalize(this);
        }
    }
}
