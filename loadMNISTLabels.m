function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);%MATLAB��~=��ʾ���Ⱥ�
%�տ�ʼ���ĸ��ֽڣ���8��16��������Ϊħ���������ж�����ļ��ǲ���MNIST�����ĳ���ļ�
magic = fread(fp, 1, 'int32', 0, 'ieee-be');%For more details about fread see http://ww2.mathworks.cn/help/matlab/ref/fread.html#btp1twt-1-machinefmt
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');%inf��ʾ�����е�������һ�У�����unsigned char�ĸ�ʽ

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end