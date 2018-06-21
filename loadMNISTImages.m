function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');%Ҳ����˵ħ������һ��4�ֽ���ͼ���������
%������8���ֽ�Ӧ���Ǳ�ʾ��ÿ��ͼ��Ĵ�С
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');%����
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');%����

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);%reshape����������䣬���������
images = permute(images,[2 1 3]);%permute��������ά�ȣ�[2 1 3]��ʾ�µ�ά��˳��
%ע��permute�������Ȱ�����䣬���ǰ����µ�ά��˳��ֱ�ӷ�ת
%����ĳ�����reshape���õ����Ȱ������ľ���֮����permute��ǰ����ά�ȶԵ����õ��൱�����Ȱ������ľ���

fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));%���ÿ��ͼ���Ӧһ����������
% Convert to double and rescale to [0,1]
images = double(images) / 255;
%images�ĵ���ά�ȱ�ʾͼƬ����������������Ӧ����һ��60000��ͼ��ÿ��ͼ����784������
end