function model = LULUPAPI(labels, K, tK, C, C_star,gamma, m)
    n       = length(labels);
    uy      = unique(labels);
    
    assert(size(K, 1) == n);
    assert(size(K, 2) == n);

    
    K   = K + 1;    % append bias
    tK  = tK + 1;   % append bias
        
    % for privileged information kernel
    H_p = tK * inv(gamma * eye(m) + C_star * tK);
    % for existing information kernel
    H_e = diag(1 ./ C ); 

    H = zeros(n,n);
    H(1:m,1:m) = H_p;
    H(m+1:n,m+1:n) = H_e;
    
    opt = ['-s 2 -t 4 -n ', num2str(1/n)]; % one class choose 2;
    if length(uy) == 2
        if uy(1)== -1 && uy(2) == 1
            uy(1) = 1;
            uy(2) = -1;    
        end
        y = -ones(n, 1);
        y(labels==uy(1)) = 1;
        Q = K.*(y*y') + H;
        % model = svmtrain(ones(n, 1), [(1:n)' Q], opt);
        model = svmtrain(y, [(1:n)' Q], opt);
    else
        model = cell(0);
        for i = 1:length(uy)
            y = -ones(n, 1);
            y(labels==uy(i)) = 1;
            Q = K.*(y*y') + H;
            % model{i} = svmtrain(ones(n, 1), [(1:n)' Q], opt);
            model{i} = svmtrain(y, [(1:n)' Q], opt);
        end
    end
    
    end
