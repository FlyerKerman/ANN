train = loadMNISTImages('train-images-idx3-ubyte');
trainLabel = loadMNISTLabels('train-labels-idx1-ubyte');
test = loadMNISTImages('t10k-images-idx3-ubyte');
testLabel = loadMNISTLabels('t10k-labels-idx1-ubyte');
a=0.0;
wait1 = waitbar(0,'׼����ʼ');
wait2 = waitbar(0,'׼����ʼ');
step = 1.55;
hnum = 200;
wh1 = 0.01*randn(784,hnum);
bh1 = 0.5*ones(1,hnum);
wh2 = 0.01*randn(hnum,hnum);
bh2 = 0.5*ones(1,hnum);
wout = 0.01*randn(hnum,10);
bout = 0.5*ones(1,10);
oneHot = [1,0,0,0,0,0,0,0,0,0;0,1,0,0,0,0,0,0,0,0;0,0,1,0,0,0,0,0,0,0;0,0,0,1,0,0,0,0,0,0;0,0,0,0,1,0,0,0,0,0;0,0,0,0,0,1,0,0,0,0;0,0,0,0,0,0,1,0,0,0;0,0,0,0,0,0,0,1,0,0;0,0,0,0,0,0,0,0,1,0;0,0,0,0,0,0,0,0,0,1];
losses = zeros(1,60000);
error = zeros(1,10000);
timecount = 0;
deltaOut = zeros(10,1);
deltaH = zeros(50,1);
%deltaWoutmae = zeros(hnum,10);
%deltaWinmae = zeros(784,hnum);
deltaWout = zeros(hnum,10);
deltaWh1 = zeros(784,hnum);
deltaWh2 = zeros(hnum,hnum);
jumpLength = 60;
jumpNum = 60000/jumpLength;
for j = 1:jumpLength
	waitbar(j/jumpLength,wait2,sprintf('net2:���ڽ��е�%d����Ծ����:%02.2f%%',j,j*100/jumpLength));
	for t = 1:jumpLength:60000
		time = mod(timecount,jumpNum);
		waitbar(time/jumpNum,wait1,sprintf('net2:��%d����Ծ������%02.2f%%',time,time*100/jumpNum));
		i = t+j-1;
		input = train(:,i)';%input��һ��������
		h1 = sigmoid(input*wh1+bh1);%h������Ǹ�������
        h2 = sigmoid(h1*wh2+bh2);
		out = sigmoid(h2*wout+bout)';%out����ת�ñ����������
		label = oneHot(:,trainLabel(i)+1);
		loss = 0.5 * sum((out-label).^2);
        
		deltaOut = out.*(1-out).*(label-out);%������
		deltaH2 = h2'.*(1-h2').*(wout*deltaOut);%������
        deltaH1 = h1'.*(1-h1').*(wh2*deltaH2);
        deltaWout = step*h2'*deltaOut';
        deltaWh2 = step*h1'*deltaH2';
        deltaWh1 = step*input'*deltaH1';
        %deltaWinnew = step*input'*deltaH';
		%deltaIn = input.*(1-input).*(win*deltaH);
		wout = wout + deltaWout;
		wh2 = wh2 + deltaWh2;
        wh1 = wh1 + deltaWh1;
        %deltaWoutmae = deltaWoutnew;%��¼����������ʷȡֵ
        %deltaWinmae = deltaWinnew;
        bout = bout + step*deltaOut';
        bh2 = bh2 + step*deltaH2';
        bh1 = bh1 + step*deltaH1';
		timecount = timecount+1;
		losses(timecount) = loss;
	end
end
close(wait1);
close(wait2);
plot(losses);
title('net2');
wait3 = waitbar(0,'׼����ʼ����');
count = 0;
results = zeros(1,10000);
for i = 1:10000
    waitbar(i/10000,wait3,sprintf('���ڽ��е�%d�β���:%02.2f%%',i,i/100));
    testInput = test(:,i);
    %targetLabel = oneHot(testLabel(i)+1);
    h1_test = sigmoid(testInput'*wh1+bh1);
    h2_test = sigmoid(h1_test*wh2+bh2);
    out_test = sigmoid(h2_test*wout+bout);
    result = find(out_test==max(out_test));
    results(i) = result;
    if(result == testLabel(i)+1)
        count = count+1;
    end
    %error(i) = 0.5*sum((out_test-targetLabel).^2);
    rate = count/10000;
end

close(wait3);