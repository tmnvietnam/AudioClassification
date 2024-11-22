private async Task StartTestHandler()
{
    while (ReadyTest)
    {
        if (BarcodeReady)
        {
            await Dispatcher.Invoke(async () =>
            {
                //int maxRetryTime = 60;
                int timePass = 0;

                //ResetMediaPlayer();

                //int idAudioTrain = -1;

                ResultTime.Content = $"{timePass}/3";

                ResultFinal.Background = new SolidColorBrush(Colors.WhiteSmoke);
                ResultFinal.Foreground = new SolidColorBrush(Colors.Black);
                ResultFinal.Content = "TESTING";

                if (File.Exists(modelFilePath) && (!EnableReTraining || Path.Exists(datasetDir)))
                {
                    bool resultSound = false;
                    for (int idxSound = 0; idxSound < 3; idxSound++)
                    {
                        try
                        {
                            //idAudioTrain++;
                            //await viewModel.StartRecordingSavingAsync(idAudioTrain);

                            if (Check())
                            {
                                //AddMediaPlayer(idAudioTrain, OxyColors.Green);
                                //MediaPlayerScrollViewer.ScrollToEnd();

                                timePass++;
                                ResultTime.Content = $"{timePass}/3";
                                resultSound = true;
                                continue;

                            }
                            else
                            {
                                //AddMediaPlayer(idAudioTrain, OxyColors.Red);
                                //MediaPlayerScrollViewer.ScrollToEnd();

                                resultSound = false;
                                break;
                            }
                        }
                        catch (Exception ex)
                        {
                            // Handle exceptions from recording or prediction
                            MessageBox.Show($"An error occurred: {ex.Message}");
                        }

                    }

                    metatTimeTotal++;

                    // Update final result based on whether any sound was successfully predicted
                    if (resultSound)
                    {
                        ResultFinal.Background = new SolidColorBrush(Colors.LawnGreen);
                        ResultFinal.Foreground = new SolidColorBrush(Colors.White);
                        ResultFinal.Content = "OK";
                        metatTimePass++;

                    }
                    else
                    {
                        ResultFinal.Background = new SolidColorBrush(Colors.Red);
                        ResultFinal.Foreground = new SolidColorBrush(Colors.White);
                        ResultFinal.Content = "NG";


                    }

                    SystemIO.Lock = false;
                    SystemIO.SendControl();


                    ResultSumary.Content = $"OK: {metatTimePass}     NG: {metatTimeTotal - metatTimePass}";

                }
                else
                {
                    if (!Path.Exists(modelFilePath))
                    {
                        MessageBox.Show("Model file not found.");
                    }

                    if (EnableReTraining && !Path.Exists(datasetDir))
                    {
                        MessageBox.Show("Dataset folder not found.");
                    }
                }


            });

            BarcodeReady = false;

            //ResultFinal.Background = new SolidColorBrush(Colors.WhiteSmoke);
            //ResultFinal.Foreground = new SolidColorBrush(Colors.Black);
            //ResultFinal.Content = "READY";

            continue;
        }

        else
        {
            continue;
        }
    }

}