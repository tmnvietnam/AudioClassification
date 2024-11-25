using System;
using System.IO;
using System.Text;
using System.Windows.Controls;

namespace SoundKit
{
    public class TextBoxStreamWriter : TextWriter
    {
        private readonly TextBox _output;

        public TextBoxStreamWriter(TextBox output)
        {
            _output = output;
        }

        public override void Write(char value)
        {
            _output.Dispatcher.Invoke(() => _output.AppendText(value.ToString()));
        }

        public override void Write(string value)
        {
            _output.Dispatcher.Invoke(() => _output.AppendText(value));
        }

        public override Encoding Encoding => Encoding.UTF8;
    }
}
