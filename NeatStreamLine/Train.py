def Train(Param):
    %run ./FNO.py
    %run ./Data_Generator/KFRDG.ipynb
################################################################
#Parameters
################################################################

    Alpha = Param["Alpha"]
    InitialSolve = Param["InitialSolve"]
    Ndt = Param["Ndt"]
    TrainingSamples = Param["TrainingSamples"] 
    ntrain = Param["ntrain"] 
    ntest = Param["ntest"]
    S = Param["S"]
    batch_size = Param["batch_size"]
    learning_rate = Param["learning_rate"]
    epochs = Param["epochs"]
    step_size = Param["step_size"]
    gamma = Param["gamma"]
    modes = Param["modes"]
    width = Param["width"]
    
    
    Udict = DataGenerator_Alpha(Alpha, TrainingSamples, Ndt, InitialSolve)   
################################################################
# read data
################################################################

    path_x = 'Data_Generator/' + f'input_{Alpha}_{InitialSolve}.npy'
    path_y = 'Data_Generator/' + f'u_results_{Alpha}_{InitialSolve}.npy'

    x_data = torch.tensor(np.load(path_x, allow_pickle=True))
    y_data = torch.tensor(np.load(path_y, allow_pickle=True))


    x_train = x_data[:ntrain,:]
    y_train = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]

    x_train = x_train.reshape(ntrain,S,1)
    x_test = x_test.reshape(ntest,S,1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    # model
    model = FNO1d(modes, width).cuda()
    #print(count_params(model))
    
################################################################
# training and evaluation
################################################################
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x)

            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward() # use the l2 relative loss

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x)
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        print(ep, t2-t1, train_mse, train_l2, test_l2)

    # torch.save(model, 'model/ns_fourier_burgers')
    pred = torch.zeros(y_test.shape)
    index = 0
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
    with torch.no_grad():
        for x, y in test_loader:
            test_l2 = 0
            x, y = x.cuda(), y.cuda()

            out = model(x).view(-1)
            pred[index] = out

            test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
            print(index, test_l2)
            index = index + 1

    print(pred.shape)
    print(y_test.shape)
    print(x_train.shape)
    #scipy.io.savemat('pred/burger_test.mat', mdict={'pred': pred.cpu().numpy()})
    
    torch.save(model, f'KFR_FNO_skiptype{Ndt}_alpha{Alpha}_trainingsamples{TrainingSamples}')
    
    return Udict