def Analyze(Param):
    %run ./FNO.py
    %run ./Data_Generator/KFRDG.ipynb
    
    %run ./SaveFig.ipynb
    Alpha = Param["Alpha"]
    InitialSolve = Param["InitialSolve"]
    Ndt = Param["Ndt"]
    TrainingSamples = Param["TrainingSamples"]
    FinalSolve = Param["FinalSolve"]
    S = Param["S"]

    Initial = Param["Initial"]
    Cutoff = Param["Cutoff"]
    
    Path_U_IS = 'Data_Generator/' + f'u_original_{Alpha}_{InitialSolve}.npy' 
    
    U_InitialSolve = torch.tensor(np.load(Path_U_IS, allow_pickle=True))
    
    model = torch.load(f'KFR_FNO_skiptype{Ndt}_alpha{Alpha}_trainingsamples{TrainingSamples}')
    
    U_Theoretical = DataGenerator_Alpha(Alpha, TrainingSamples, Ndt, FinalSolve)
    
    U_Gt = U_Theoretical['u_original']
    Time = U_Theoretical['time'][-1]
    
    L = 6
    N = 200
    dx = L / N  # grid size

    x = np.empty(N + 7)
    x[0:3] = [(-L + dx * i) for i in [-3, -2, -1]]  # left ghost points
    x[3:N + 4] = [(-L + dx * i) for i in range(N + 1)]  # internal grid points on [-L,0]
    x[N + 4:N + 7] = np.arange(1, 4) * dx  # grid points ahead of the shock

    Length=np.array([x])[-1]
    
    EndForTime = (Time[::Ndt].shape[-1]-U_InitialSolve[::Ndt,-1].shape[-1])-1
    
    #####
    #Main Loop
    #####
    time = []
    w = model(U_InitialSolve[-1,:].float().reshape(1,S,1).cuda())
    time.append(w.cpu().detach().numpy()) 
    for i in range(0,EndForTime):
        with torch.no_grad():
            w = model(w)
        time.append(w.cpu().detach().numpy())

    timenp = np.array(time)

    time_shock = timenp[:,-1,:,-1]
    time_pde = timenp[:,-1,:,:]
    
    Inf, L2 = FigureSave(Time,Length,U_Gt,U_InitialSolve,time_pde,time_shock,Ndt,Initial,Cutoff,Alpha)
    
    return Inf, L2