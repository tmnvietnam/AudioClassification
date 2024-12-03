using System.Collections;
using System.Runtime.CompilerServices;
using System.Windows.Shapes;

namespace SoundKit.Serial
{
    public class SystemIO : Device
    {

        public string Title;

        public SystemIO(string title, Rectangle connectionStatusDevice, Rectangle txStatusDevice, Rectangle rxStatusDevice) : base(title, connectionStatusDevice, txStatusDevice, rxStatusDevice)
        {
           
        }

        public event EventHandler OnStartRequest;


        // state define
        public const bool ON = true;
        public const bool OFF = false;

        // Machine SYSTEM OUTPUT
        /// <summary>
        /// Software sw control main cylinder going up
        /// </summary>
        private bool _ResetOUT;

        public bool ResetOUT
        {
            get { return _ResetOUT; }
            set
            {
                if (_ResetOUT != value)
                {
                    _ResetOUT = value;
                }
 
            }
        }

        private bool _ResetIN;

        public bool ResetIN
        {
            get { return _ResetIN; }
            set
            {
                if (_ResetIN != value)
                {
                    _ResetIN = value;
                }

            }
        }

        private bool _StartIN;

        public bool StartIN
        {
            get { return _StartIN; }
            set
            {
                if (_StartIN != value)
                {
                    _StartIN = value;
                    if (_StartIN == ON)
                    {
                        OnStartRequest?.Invoke(null, null);
                    }
            
                }

            }
        }


        public void SendControl()
        {
            var data = DataSending();
            if (SerialPort.IsOpen)
            {
                SendBytes(CreateFrame(data));
            }
        }

        public void DataReceived(byte[] bytes)
        {
            if (bytes.Length != 4)
            {
                return;
            }
            uint Data32Bit = BitConverter.ToUInt32(bytes, 0);

            //InputIO
            ResetIN = GetValue(Data32Bit, 8);
            StartIN = GetValue(Data32Bit, 9);
            Console.WriteLine();
        }

        public bool GetValue(uint data, int position)
        {
            return (data & (uint)(1 << position)) != 0;
        }

        public byte[] DataSending()
        {
            byte[] bytes = new byte[5];
            bytes[0] = 0x4F;
            List<bool> OutPuts = new List<bool>
            {
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                false,

                false,
                false,
                false,
                false,
                false,
                false,
                false,
                false,

                false,
                false,
                false,
                false,
                false,
                false,
                false,
                false,

                ResetOUT,
                false,
                false,
                false,
                false,
                false,
                false,
                false
            };

            BitArray bits = new BitArray(OutPuts.ToArray());
            bits.CopyTo(bytes, 1);
            return bytes;
        }


    }

}
