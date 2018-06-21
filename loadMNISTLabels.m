function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);%MATLAB中~=表示不等号
%刚开始的四个字节（共8个16进制数）为魔数，用于判断这个文件是不是MNIST里面的某个文件
magic = fread(fp, 1, 'int32', 0, 'ieee-be');%For more details about fread see http://ww2.mathworks.cn/help/matlab/ref/fread.html#btp1twt-1-machinefmt
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');%inf表示把所有的数读成一列，按照unsigned char的格式

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end