wait3 = waitbar(0,'准备开始测试');
count = 0;
results = zeros(1,10000);
targets = [1 5 0 3 1 1 3 8];
j=1;
targetImages = zeros(28,28,8);
targetIndex = zeros(1,8);
for i = 1:60000
    waitbar(j/8,wait3,sprintf('正在寻找第%d个目标数字',j+1));
    if(trainLabel(i)==targets(j))
        testInput = train(:,i);
        %targetLabel = oneHot(testLabel(i)+1);
        h_test = sigmoid(testInput'*win+bin);
        out_test = sigmoid(h_test*wout+bout);
        %result = find(out_test==max(out_test));
        targetImages(:,:,j) = reshape(out_test,28,28);
        targetIndex(j)=i;
        j=j+1;
    end
    if(j>=9)
        break;
    end
    %results(i) = result;
    %if(result == testLabel(i)+1)
       % count = count+1;
    %end
    %error(i) = 0.5*sum((out_test-targetLabel).^2);
    %rate = count/10000;
end
figure(2);
for i = 1:4
    subplot(4,2,2*i-1);
    imagesc(reshape(train(:,targetIndex(i)),28,28));
    subplot(4,2,2*i);
    imagesc(targetImages(:,:,i));
end
figure(3);
for i = 5:8
    subplot(4,2,2*(i-4)-1);
    imagesc(reshape(train(:,targetIndex(i)),28,28));
    subplot(4,2,2*(i-4));
    imagesc(targetImages(:,:,i));
end
close(wait3);