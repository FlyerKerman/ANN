train = loadMNISTImages('train-images-idx3-ubyte');
trainLabel = loadMNISTLabels('train-labels-idx1-ubyte');
test = loadMNISTImages('t10k-images-idx3-ubyte');
testLabel = loadMNISTLabels('t10k-labels-idx1-ubyte');
a=0.0;
wait1 = waitbar(0,'准备开始');
wait2 = waitbar(0,'准备开始');
step = 0.1;
hnum = 100;
win = 0.01*randn(784,hnum);
%bin = 0.01*randn(1,hnum);
bin = 0.5*ones(1,hnum);
wout = 0.01*randn(hnum,10);
bout = 0.5*ones(1,10);
oneHot = [1,0,0,0,0,0,0,0,0,0;0,1,0,0,0,0,0,0,0,0;0,0,1,0,0,0,0,0,0,0;0,0,0,1,0,0,0,0,0,0;0,0,0,0,1,0,0,0,0,0;0,0,0,0,0,1,0,0,0,0;0,0,0,0,0,0,1,0,0,0;0,0,0,0,0,0,0,1,0,0;0,0,0,0,0,0,0,0,1,0;0,0,0,0,0,0,0,0,0,1];
losses = zeros(1,60000);
error = zeros(1,10000);
timecount = 0;
deltaOut = zeros(10,1);
deltaH = zeros(hnum,1);
deltaWoutmae = zeros(hnum,10);
deltaWinmae = zeros(784,hnum);
deltaWoutnew = zeros(hnum,10);
deltaWinnew = zeros(784,hnum);
jumpLength = 60;
jumpNum = 60000/jumpLength;
for j = 1:jumpLength
	waitbar(j/jumpLength,wait2,sprintf('net1:正在进行第%d组跳跃采样:%02.2f%%',j,j*100/jumpLength));
	for t = 1:jumpLength:60000
		time = mod(timecount,jumpNum);
		waitbar(time/jumpNum,wait1,sprintf('net1:第%d次跳跃采样：%02.2f%%',time,time*100/jumpNum));
		i = t+j-1;
		input = train(:,i)';%input是一个行向量
		h = sigmoid(input*win+bin);%h算出来是个行向量
		out = sigmoid(h*wout+bout)';%out经过转置变成了列向量
		label = oneHot(:,trainLabel(i)+1);
		loss = 0.5 * sum((out-label).^2);
        
		deltaOut = out.*(1-out).*(label-out);%列向量
		deltaH = h'.*(1-h').*(wout*deltaOut);%列向量
        deltaWoutnew = step*h'*deltaOut';
        deltaWinnew = step*input'*deltaH';
		%deltaIn = input.*(1-input).*(win*deltaH);
		wout = wout + a*deltaWoutmae + deltaWoutnew ;
		win = win + a*deltaWinmae + deltaWinnew;
        deltaWoutmae = deltaWoutnew;%记录调整量的历史取值
        deltaWinmae = deltaWinnew;
        bout = bout + step*deltaOut';
        bin = bin + step*deltaH';
		timecount = timecount+1;
		losses(timecount) = loss;
	end
end
close(wait1);
close(wait2);
plot(losses);
title('net1');
wait3 = waitbar(0,'准备开始测试');
count = 0;
results = zeros(1,10000);
for i = 1:10000
    waitbar(i/10000,wait3,sprintf('正在进行第%d次采样:%02.2f%%',i,i/100));
    testInput = test(:,i);
    %targetLabel = oneHot(testLabel(i)+1);
    h_test = sigmoid(testInput'*win+bin);
    out_test = sigmoid(h_test*wout+bout);
    result = find(out_test==max(out_test));
    results(i) = result;
    if(result == testLabel(i)+1)
        count = count+1;
    end
    %error(i) = 0.5*sum((out_test-targetLabel).^2);
    rate = count/10000;
end

close(wait3);