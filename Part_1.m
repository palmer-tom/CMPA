%
% ELEC 4700 PA 8
%
% Diode Paramater Extraction
%
% Tom Palmer - 101045113
%
% 06 MAR 2020
%

% Setup
clear
clc

% Define Variables
V = linspace(-1.95,0.7,200);
Is = 0.01e-12;
Ib = 0.1e-12;
Vb = 1.3;
Gp = 0.1;

I = zeros(1,200);
In = zeros(1,200);

for n = 1:200
    
    I(1,n) = (Is*(exp(V(1,n)*1.2/0.025)-1)) + (Gp*V(1,n)) - Ib*(exp(1.2*(-(V(1,n)+Vb))/0.025)-1);
    
end

noise = (rand(1,200)*0.4)+0.8;

for n = 1:200
    
    In(1,n) = noise(1,n)*I(1,n);
    
end

figure(1)
plot(V,I)
title("No Noise - 4th Order");
hold on
pI_4th = polyfit(V,I,4);
I_4th = polyval(pI_4th,V);
plot(V,I_4th)
hold off

figure(2)
plot(V,I)
title("No Noise - 8th Order");
hold on
pI_8th = polyfit(V,I,8);
I_8th = polyval(pI_8th,V);
plot(V,I_8th)
hold off

figure(3)
semilogy(V,abs(I))
title("No Noise - Log Scale");
hold on
semilogy(V,abs(I_8th))
hold off

figure(4)
plot(V,In)
title("Noise - 4th Order");
hold on
pIn_4th = polyfit(V,In,4);
In_4th = polyval(pIn_4th,V);
plot(V,In_4th)
hold off

figure(5)
plot(V,In)
title("Noise - 8th Order");
hold on
pIn_8th = polyfit(V,In,8);
In_8th = polyval(pIn_8th,V);
plot(V,In_8th)
hold off

figure(6)
semilogy(V,abs(In))
title("Noise - Log Scale");
hold on
semilogy(V,abs(In_8th))
hold off

% Using fit()

% fit A & C
fo = fittype('(A*(exp(x*1.2/0.025)-1)) + (Gp*x) - C*(exp(1.2*(-(x+Vb))/0.025)-1)');
ff = fit(V',I',fo, 'StartPoint', [0,0,0,0]);
If = ff(V);
figure(7)
plot(V,If)
title("fit A & C");


% fit A, B, & C
fo = fittype('(A*(exp(x*1.2/0.025)-1)) + (B*x) - C*(exp(1.2*(-(x+Vb))/0.025)-1)');
ff = fit(V',I',fo, 'StartPoint', [0,0,0,0]);
If = ff(V);
figure(8)
plot(V,If)
title("fit A, B, & C");

% fit A, B, C, & D
fo = fittype('(A*(exp(x*1.2/0.025)-1)) + (B*x) - C*(exp(1.2*(-(x+D))/0.025)-1)');
ff = fit(V',I',fo, 'StartPoint', [0,0,0,0]);
If = ff(V);
figure(9)
plot(V,If)
title("fit A, B, C, & D");



% Using neural net model
inputs = V.';
targets = I.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net)
Inn = outputs

figure(10)
plot(V,Inn)
title('Neural Net Output')

% Using neural net model
inputs = V.';
targets = In.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net)
Inn = outputs

figure(10)
plot(V,Inn)
title('Neural Net Output Noise')
