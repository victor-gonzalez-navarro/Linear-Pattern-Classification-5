
%% PRÁCTICA 6, BD: Brain
clear
%close all
clc 
load PRBB_Brain
i_dib=1;                     % 1 Dibuja las imágenes
i_clas_NN=1;                 % 1 Clasifica Imagen completa mediante NN
i_clas_Tree=0;               % 1 Clasifica Imagen completa mediante Tree
i_valida_hidden=0;           % 1 Valida número neuronas en capa oculta
i_valida_split=0;            % 1 Valida número máximo de splits in trees
rng('shuffle') 

%% Se dibujan opcionalmente las imágenes
N_images=8;
Ndim=256;
N_datos=Ndim*Ndim;
Brain_8=zeros(N_datos,N_images);
for i1=1:N_images
    Vaux=Brain(:,i1);
    Aux=reshape(Vaux,Ndim,Ndim);
    if i_dib==1
        figure
        imagesc(Aux)
        axis image
        colorbar
    end
    Brain_8(:,i1)=Aux(:);
end
clear Vaux Aux N_images i1 Brain

%%  Se etiquetan los vectores (píxeles) utilizando las probabilidades de las clases (images 6,7,8)
N_feat=5;
Brain_5=Brain_8(:,1:N_feat);

% Detección de píxeles de fondo 'clase 4'
[class,ind]=max(Brain_8(:,6:8),[],2);
Index_clase0=find(class==0);
Brain_Etiq(1:length(Index_clase0),1:N_feat)=Brain_5(Index_clase0,1:N_feat);
Labels(1:length(Index_clase0))=4*ones(length(Index_clase0),1);

%Detección del resto de píxeles etiquetados
Pr_min=0.9;
Index_Labels=find(class>=Pr_min);
Brain_Etiq = [Brain_Etiq ; Brain_5(Index_Labels,1:N_feat)];
Labels(length(Index_clase0)+1: length(Index_clase0)+length(Index_Labels))=ind(Index_Labels);
%
if i_dib==1
    %Representación de BD Etiquetada.
    Labeled_Image=zeros(N_feat,1);
    Labeled_Image(Index_clase0)=4;
    Labeled_Image(Index_Labels)=ind(Index_Labels);
    Aux=reshape(Labeled_Image,Ndim,Ndim);
    figure('name','Labeled Image')
    imagesc(Aux)
    axis image
    colorbar
    clear Labeled_Image Aux class ind
end

clear Brain_8 class ind i_dib

%% BD
% Eliminar excedentes de class=4 (Background')
N_classes=4;
N_size=zeros(1,N_classes);
Brain_Etiq2=[];
Labels2=[];
for i_class=1:N_classes-1
    N_size(i_class)=length(find(Labels==i_class));
    Brain_Etiq2=[Brain_Etiq2;Brain_Etiq(find(Labels==i_class),:)];
    Labels2=[Labels2 i_class*ones(1,N_size(i_class))];
end
N_c4=round(mean(N_size(1:N_classes-1))); %Reduccion de numero elementos clase fondo
Vaux=find(Labels==N_classes);
Brain_Etiq2=[Brain_Etiq2;Brain_Etiq(Vaux(1:N_c4),:)];
Labels2=[Labels2 N_classes*ones(1,N_c4)]; %Nuevo vector de etiquetas
clear Vaux N_c4 N_size indexperm i_class
Brain_Etiq=Brain_Etiq2;
Labels=Labels2;
clear Labels2 Brain_Etiq2 Index_clase0 Index_Labels %Index_NO_label

%Aleatorización orden de los vectores
Permutation=randperm(length(Labels));
Brain_Etiq=Brain_Etiq(Permutation,:);
Labels=Labels(Permutation);

%% Generación Índices de BD Train, BD Val, BD Test
P_train=0.6;
P_val=0.2;
P_test=1-P_train-P_val;
Index_train=[];
Index_val=[];
Index_test=[];
Labels=Labels';
for i_class=1:N_classes
    index=find(Labels==i_class);
    N_i_class=length(index);
    [I_train,I_val,I_test] = dividerand(N_i_class,P_train,P_val,P_test);
    Index_train=[Index_train;index(I_train)];
    Index_val=[Index_val;index(I_val)];
    Index_test=[Index_test;index(I_test)];
end
% Mixing of vectors not to have all belonging to a class together
Permutation=randperm(length(Index_train));
Index_train=Index_train(Permutation);
Permutation=randperm(length(Index_val));
Index_val=Index_val(Permutation);
Permutation=randperm(length(Index_test));
Index_test=Index_test(Permutation);
clear Permutation i_class index N_i_class I_train I_val I_test

% Generación BD Train, BD CV, BD Test
X_train=Brain_Etiq(Index_train,:);
Labels_train=Labels(Index_train);
X_val=Brain_Etiq(Index_val,:);
Labels_val=Labels(Index_val);
X_test=Brain_Etiq(Index_test,:);
Labels_test=Labels(Index_test);

%% NEURAL NETWORKS
if i_clas_NN==1
    % net output target
    Target_BD=zeros(N_classes,length(Labels));
    for i_element=1:length(Labels)
        Target_BD(Labels(i_element),i_element)=1;
    end
    clear i_element
    % Create a Pattern Recognition Network
    hiddenLayerSize = 10;
    net = patternnet([hiddenLayerSize,]);
    net.performFcn='mse';
    %net.trainFcn='trainscg';  % Conjugate gradient
    %net.trainFcn='traingd';   %Back Propagation
    net.trainFcn='trainlm';   %Levenberg-Marquadt
    net = configure(net,Brain_Etiq',Target_BD);
    
    net.divideFcn='divideind'; % The database is divided by indices
    net.divideParam.trainInd=Index_train;
    net.divideParam.valInd=Index_val;
    net.divideParam.testInd=Index_test;
    
    net.trainParam.epochs = 10000;
    %net.trainParam.max_fail=round(net.trainParam.epochs/10); % Can set the
    %number of consecutive high values of the error over epochs in the validation set.
    %Used to stop the training.
    
    net = train(net,Brain_Etiq',Target_BD);% Train the Network
    % Measure Train error
    outputs = net(X_train');
    [~, Index_out]=max(outputs);
    NN_Error_train=length(find(Labels_train~=Index_out'))/length(Labels_train);
    fprintf(1,' error NN train = %g   \n', NN_Error_train);
    CM_Train=confusionmat(Labels_train,Index_out)
    % Measure val error
    outputs = net(X_val');
    [~, Index_out]=max(outputs);
    NN_Error_val=length(find(Labels_val~=Index_out'))/length(Labels_val);
    fprintf(1,' error NN val = %g   \n', NN_Error_val);
    CM_Val=confusionmat(Labels_val,Index_out)
    % Measure Test error
    outputs = net(X_test');
    [~, Index_out]=max(outputs);
    NN_Error_test=length(find(Labels_test~=Index_out'))/length(Labels_test);
    fprintf(1,' error NN test = %g   \n', NN_Error_test);
    CM_Test=confusionmat(Labels_test,Index_out)
    %% NEURAL NETWORKS IMAGE CLASSIFICATION
    outputs = net(Brain_5');
    [~, Index_out]=max(outputs);
    Aux=reshape(Index_out,Ndim,Ndim);
    figure('name','NN Classified Image')
    imagesc(Aux)
    axis image
    colorbar
    clear Index_out Aux
end
clear i_clas_NN

%% Validacion de número de neuronas en capa oculta
if i_valida_hidden==1;
    
    % TO DO
    
    % Generate Train, Val & Test BDs and Labels
    % Prepair Train targets
    % for hiddenLayerSize =d:3d
    % The net percentages for validation and test are 0% using the code:
%             net.divideParam.trainRatio = 1;
%             net.divideParam.valRatio = 0;
%             net.divideParam.testRatio = 0;
  % Train a net (LM) using the train BD and the train targets
    % Measure Train, Val and Test classification errors using:
    %   outputs = net(BD');
    % end for
    % PLot train, val and test errors with the number of hidden neurons
    
    % END TO DO
    
end
clear i_valida_hidden
