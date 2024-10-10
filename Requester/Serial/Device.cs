
using System.IO.Ports;
using System.Windows.Media;
using System.Windows.Shapes;
using System.Windows.Threading;
using Timer = System.Timers.Timer;


namespace SoundKit.Serial
{
    public class Device
    {
        public enum CheckSumType
        {
            XOR,
            CRC8,
            CRC16,
            CRC16_CCITT,
            CRC16_MOSBUS,
            CRC32,
            CRC8_REVERSED,
            SUM
        }

        public static byte Prefix1 = 0x44;
        public static byte Prefix2 = 0x45;
        public static byte Suffix = 0x56;

        public SerialPort SerialPort;

        public CancellationTokenSource _shutDown = new CancellationTokenSource();
        public event EventHandler SerialReceiverHandler;


        private Rectangle connectionStatusDevice;
        private Rectangle txStatusDevice;
        private Rectangle rxStatusDevice;

        private Timer rxTimer = new Timer()
        {
            Interval = 25,
        };

        private Timer txTimer = new Timer()
        {
            Interval = 25,
        };


        public List<int> Buffer = new List<int>();

        public Device(Rectangle connectionStatusDevice, Rectangle txStatusDevice, Rectangle rxStatusDevice)
        {
            SerialPort = new SerialPort()
            {
                BaudRate = 9600,
                ReadTimeout = 500,
            };
            this.connectionStatusDevice = connectionStatusDevice;
            this.txStatusDevice = txStatusDevice;
            this.rxStatusDevice = rxStatusDevice;

            txTimer.Elapsed += TxTimer_Tick;
            rxTimer.Elapsed += RxTimer_Tick;

            txTimer.Enabled = true;
            rxTimer.Enabled = true;
        }

        private void RxTimer_Tick(object sender, EventArgs e)
        {
            if (!_shutDown.IsCancellationRequested)
            {
                rxStatusDevice.Dispatcher.Invoke(new Action(() =>
            {
                rxStatusDevice.Fill = new SolidColorBrush(Color.FromRgb(198, 198, 198));
            }), DispatcherPriority.Normal);
                rxTimer.Stop();
            }
        }

        private void TxTimer_Tick(object sender, EventArgs e)
        {
            if (!_shutDown.IsCancellationRequested)
            {
                txStatusDevice.Dispatcher.Invoke(new Action(() =>
                {
                    txStatusDevice.Fill = new SolidColorBrush(Color.FromRgb(198, 198, 198));
                }), DispatcherPriority.Normal);
                txTimer.Stop();
            }
        }

        private void Port_DataReceived(object sender, SerialDataReceivedEventArgs e)
        {
            if (!_shutDown.IsCancellationRequested)
            {
                rxStatusDevice.Dispatcher.Invoke(new Action(() =>
            {
                rxStatusDevice.Fill = new SolidColorBrush(Color.FromRgb(117, 247, 17));
                rxTimer.Start();
            }));
            }


            SerialReceiverHandler?.Invoke(sender, null);

        }

        public static byte CalculateChecksum(byte[] byteData) //Dis
        {
            Byte chkSumByte = 0x00;
            for (int i = 0; i < byteData.Length; i++)
                chkSumByte ^= byteData[i];
            return chkSumByte;
        }

        public static byte CreateChecksum(byte[] bytes, CheckSumType type)
        {
            switch (type)
            {
                case CheckSumType.XOR:
                    return CalculateChecksum(bytes);
                case CheckSumType.CRC8:
                    break;
                case CheckSumType.CRC16:
                    break;
                case CheckSumType.CRC16_CCITT:
                    break;
                case CheckSumType.CRC16_MOSBUS:
                    break;
                case CheckSumType.CRC32:
                    break;
                case CheckSumType.CRC8_REVERSED:
                    break;
                case CheckSumType.SUM:
                    break;
                default:
                    break;
            }
            return 0x00;
        }

        public static byte[] CreateFrame(byte[] datas, bool IsNoSize = false)
        {

            if (datas == null) return null;

            List<byte> dataToSend = datas.ToList();
            if (!IsNoSize)
            {
                if (datas.Length > 1)
                {
                    dataToSend.Insert(0, (byte)(dataToSend.Count + 1));
                }
                else
                {
                    dataToSend.Add(0x00);
                }
            }
            dataToSend.Insert(0, Prefix2);
            dataToSend.Insert(0, Prefix1);
            var checksum = CreateChecksum(dataToSend.ToArray(), CheckSumType.XOR);
            dataToSend.Add(checksum);
            dataToSend.Add(Suffix);

            foreach (var item in dataToSend)
            {
                Console.Write(item.ToString("X2") + " ");
            }
            Console.WriteLine(" ");
            return dataToSend.ToArray();
        }
        public void SendBytes(byte[] buf)
        {
            try
            {
                txStatusDevice.Dispatcher.Invoke(new Action(() =>
                {
                    txStatusDevice.Fill = new SolidColorBrush(Colors.Yellow);
                    txTimer.Start();
                }), DispatcherPriority.Normal);

                SerialPort.DiscardInBuffer();

                SerialPort.Write(buf, 0, buf.Length);



            }
            catch (Exception)
            {
            }
        }

        public bool Send(byte[] data)
        {
            Buffer.Clear();

            if (SerialPort.IsOpen)
            {
                try
                {
                    SendBytes(data);
                    return true;
                }
                catch (System.IO.IOException)
                {
                    return false;
                }
                catch (TimeoutException)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }

        public string ReadLine()
        {
            if (SerialPort.IsOpen)
            {
                try
                {
                    string str = SerialPort.ReadLine();
                    return str;
                }
                catch (TimeoutException)
                {
                    return "ERROR";
                }
                catch
                {
                    return "ERROR";
                }
            }
            return "ERROR";
        }


        public void CheckCommunication(string portName)
        {
            try
            {
                SerialPort.PortName = portName;

                if (SerialPort.IsOpen)
                {
                    SerialPort.Close();
                }

                SerialPort.Open();


                if (!SerialPort.IsOpen)
                {
                    connectionStatusDevice.Fill = new SolidColorBrush(Color.FromRgb(193, 193, 193));
                    SerialPort.DataReceived -= Port_DataReceived;
                }
                else
                {
                    connectionStatusDevice.Fill = new SolidColorBrush(Color.FromRgb(117, 247, 17));
                    SerialPort.DataReceived += Port_DataReceived;
                }

            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }
        }
    }
}
