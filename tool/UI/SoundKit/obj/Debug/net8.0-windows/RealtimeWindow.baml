   M S B A M L   `   `   `  ����C  ?SoundKit, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null#clr-namespace:SoundKitSoundKit  5   	       SoundKit.RealtimeWindow   5   	   W SPresentationCore, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35e aSystem.Windows.Controls.Ribbon, Version=8.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089R NWindowsBase, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35^ ZWindowsFormsIntegration, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35H 9http://schemas.microsoft.com/winfx/2006/xaml/presentation      5   	   Bd2http://schemas.microsoft.com/expression/blend/2008      5   	   Lmc;http://schemas.openxmlformats.org/markup-compatibility/2006      5   	   R NSystem.Xaml, Version=8.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5   	   R NOxyPlot.Wpf, Version=2.2.0.0, Culture=neutral, PublicKeyToken=75e952ba404cdbb0Y UOxyPlot.Wpf.Shared, Version=2.2.0.0, Culture=neutral, PublicKeyToken=75e952ba404cdbb0"oxyhttp://oxyplot.org/wpf   6@   X	 TFontAwesome.Sharp, Version=6.6.0.0, Culture=neutral, PublicKeyToken=d16d1e4e568ec10fFfa=http://schemas.awesome.incremented/wpf/xaml/fontawesome.sharp 	 5   	   "localclr-namespace:SoundKit 	 5   	     � Title$  Realtime��5	   	   $
��1000��6   $
��1700��6(   .��5      � $
��1000��6   $��Stretch=�6    � RowDefinitions 5   
   �� 5      $��1* �6   �� 5      $��1* �6   �� 5      $��40 �6   .+�5   
     OxyPlot.Wpf.PlotView  $
��1500��6   $��0��6$     OxyPlot.Wpf.PlotViewBase   Model 61   �� *PlotModelRealtimeTime+  5   
   $
��1500��6   $��1��6$    61   �� *PlotModelRealtimeFreq+�� 5   
   $��2��6   $��0��6   $��Right=�6.   $��Stretch=�6J   ���#0087DC �� CornerRadius$ 20��6{   $��40��6�   $��40��6�   $��0,0,5,0q�6�   ._�5      �� -   5        $��0q�5      
������ .��5      ! 	 FontAwesome.Sharp.IconImage  $��32��6   $��32��6+   � 	 �FontAwesome.Sharp.IconImageBase`1[[FontAwesome.Sharp.IconChar, FontAwesome.Sharp, Version=6.6.0.0, Culture=neutral, PublicKeyToken=d16d1e4e568ec10f]]   Icon$
 Redo=�66      
Foreground
 �����6B   5      5      5      5      