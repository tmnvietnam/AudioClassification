   M S B A M L   `   `   `  ����C  ?SoundKit, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null#clr-namespace:SoundKitSoundKit  5   	       SoundKit.MainWindow   5   	   W SPresentationCore, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35e aSystem.Windows.Controls.Ribbon, Version=8.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089R NWindowsBase, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35^ ZWindowsFormsIntegration, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35H 9http://schemas.microsoft.com/winfx/2006/xaml/presentation      5   	   R NSystem.Xaml, Version=8.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5   	   R NOxyPlot.Wpf, Version=2.2.0.0, Culture=neutral, PublicKeyToken=75e952ba404cdbb0Y UOxyPlot.Wpf.Shared, Version=2.2.0.0, Culture=neutral, PublicKeyToken=75e952ba404cdbb0"oxyhttp://oxyplot.org/wpf   6@   <d2http://schemas.microsoft.com/expression/blend/2008   5   	   X	 TFontAwesome.Sharp, Version=6.6.0.0, Culture=neutral, PublicKeyToken=d16d1e4e568ec10fFfa=http://schemas.awesome.incremented/wpf/xaml/fontawesome.sharp 	 5   	   Dmc;http://schemas.openxmlformats.org/markup-compatibility/2006 	 5   	   "localclr-namespace:SoundKit 	 5   	     � Title$  SoundKit��5	      $
��1050��6   $	��550��6,    � 
ResizeMode$ NoResize=�5
      .��5      � $��Center=�6   $	��Top=�6)    � RowDefinitions 5   
   �� 5      $	��120 �6   �� 5      .+�5   
   �� $��0��6   $��Center=�6"   $��Center=�6?   $��5q�6Z   $	��110��6e   .+�5      �� $��20��6   $	��480��6$   $
��Bold"�60   Open Fixture Window6W   �� 5      $
��Left=�6   $	��Top=�64    �� Orientation$ 
Horizontal=�6N   .+�5      �� -   5         ��Name$ OpenFixture1��5      $��80��6N   $��30��6Y   $��15(�6e   $
��Bold"�6s   .��6�   	Fixture 16�   �� 5      -   5         ��Name$ MicrophoneComboBox1��5      $��30��68    �� DisplayMemberPath$ ProductName��6D   $	��400��6d   $��15(�6p   5      �� 5      $
��Left=�6   $	��Top=�64   $ 
Horizontal=�6N   .+�5      �� -   5        $ OpenFixture2��5      $��80��6N   $��30��6Y   $��15(�6e   $
��Bold"�6s   .��6�   	Fixture 26�   �� 5      -   5        $ MicrophoneComboBox2��5      $��30��68   $ ProductName��6D   $	��400��6d   $��15(�6p   5      �� 5      $
��Left=�6   $	��Top=�64   $ 
Horizontal=�6M   .+�5       �� -   5        $ OpenFixture3��5       $��80��6N   $��30��6Y   $��15(�6e   $
��Bold"�6s   .��6�   	Fixture 36�   �� 5!      -   5        $ MicrophoneComboBox3��5!      $��30��68   $ ProductName��6D   $	��400��6d   $��15(�6p   5"      5#      �� 5$   
   $��1��6   $��5q�6"   $ Vertical=�6.   $��Center=�6F   $��Center=�6c   $	��820��6~   .+�5%      �� $��20��6   $	��480��6$   $
��Bold"�60   	Logging6K   �� 5&      -   5        $��	MyTextBox��5&      " �� VerticalScrollBarVisibility$
 Auto=�6'   $	��800��6J   $	��480��6X   
���   �
������$ �� HorizontalScrollBarVisibility$
 Auto=�5'      	 �� 
IsReadOnly$
	 True��6:   
 �� TextWrapping$

 Wrap=�5(      5*      5,      5.      