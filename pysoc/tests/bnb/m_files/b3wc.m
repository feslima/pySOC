function [B,sset,ops,ctime,flag]=b3wc(G1,Gd1,Wd,Wn,Juu,Jud,tlimit,nc)
% B3WC      Bidirectional Branch and Bound (B3) for Worst Case Criterion
%
%   [B,sset,ops,ctime,flag] = b3wc(G1,Gd1,Wd,Wn,Juu,Jud,tlimit,nc) is a B3
%   implementation to select measurement subsets, which have the minimum
%   worst-case loss in terms of self-optimizing control based on the
%   following static optimization problem:
%
%       min J = f(y,u,d)
%       s.t.
%           y = G1 * u + Gd1 * Wd * d + Wn * e
%
%  where y is measurement vector (ny), u is input vector (nu), d is disturance
%  vector (nd) and e is the control error vector (ny).
%
%  The worst case loss is defined based on the assumption that 
%
%       ||d'  e'||_2 = 1 
%
% Inputs:
%   G1 = process model
%   Gd1 = disturbance model
%   Wd  = diagonal matrix containing magnitudes of disturbances
%   Wn = diagonal matrix containing magnitudes of implementation errors
%   Juu = \partial^2 J/\partial u^2
%   Jud = \partial^2 J/\partial u\partial d
%   tlimit defines the time limit to run the code. Default is Inf.
%   nc determines the number of best subsets to be selected. Default is 1.
%
% Outputs:
%   B     - nc x 1 vector of the wc loss of selected subsets.
%   sset  - nc x nu indices of selected subsets
%   ops   - number of nodes evaluated.
%   ctime - computation time used
%   flag  - 0 if successful, 1 otherwise (for tlimit < Inf).
%
% References:
%  [1]    I. J. Halvorsen, S. Skogestad, J. C. Morud, and V. Alstad.
%  Optimal selection of controlled variables. Ind. Eng. Chem. Res.,
%  42(14):3273{3284, 2003.   
%  [2]   V. Kariwala, Y. Cao, and S. Janardhanan. Local self-optimizing 
%  control with average loss minimization. Ind. Eng. Chem. Res., 
%  47(4):1150-1158, 2008.
%  [3]    V. Kariwala and Y. Cao, Bidirectional Branch and Bound for
%  Controlled Variable Selection: Part II. Exact Local Method for
%  Self-optimizing Control, Computers and Chemical Engineering, 
%  33(8):1402:1412, 2009 
%
%  See also b3msv, pb3wc, randcase

%  By Yi Cao at Cranfield University, 8th January 2009
%

% Example
%{
ny=30;
nu=15;
nd=5;
[G,Gd,Wd,Wn,Juu,Jud] = randcase(ny,nu,nd);
[B,sset,ops,ctime] = b3wc(G,Gd,Wd,Wn,Juu,Jud);
% It takes about 2 seconds.
%}
%

    flag=false;
% Defaults    
    if nargin<8
        nc=1;
    end
    if nargin<7
        tlimit=Inf;
    end
% Initialization    
    ctime0=cputime;
    [r,n] = size(G1);
    % way to perform bidirection branching tuned for different size
    if (r-n)*(r-n) > n*n*n
        mode = 0;
    else
        mode = 1;
    end
    if r<50
        ns=n+1;
        ms=r-n+1;
    elseif 2*n <= r,
        ns=n+1;
        ms=5;
    else
        ns=5;
        ms=r-n+1;
    end
    % prepare matrices
    Y = [(G1*inv(Juu)*Jud-Gd1)*Wd Wn];
    G = G1*inv(sqrtm(Juu));
    Y2=Y*Y';
    Yinv=Y2;
    Gd=G;
    Xd=Y2;
    q2=diag(G*G')./diag(Y2);
    p2=q2;
    s2=p2;
    ops=zeros(1,4);
    B=zeros(nc,1);
    sset=zeros(nc,n);
    ib=1;
    bound=0;
    % counters: 1) terminal; 2) nodes; 3) sub-nodes; 4) calls
    ops=[0 0 0 0];
    fx=false(1,r);
    rem=true(1,r);
    downV=fx;
    downR=fx;
    nf=0;
    n2=n;
    m2=r;
    % initial flags
    f=false;
    bf=false;
    downf=false;
    upf=false;
    % recursive solver
    bbL2sub(fx,rem);
    % convert the bound to the worst case loss
    [B,idx]=sort(0.5./B);
    sset=sort(sset(idx,:),2);
    ctime=cputime-ctime0;
    
    function bn=bbL2sub(fx0,rem0)
        bn=0;
        if cputime-ctime0>tlimit
            flag=true;
            return;
        end        
        ops(4)=ops(4)+1;
        fx=fx0;             %fixed set
        rem=rem0;           %candidate set
        nf=sum(fx);         
        m2=sum(rem);
        n2=n-nf;
        while ~f && m2>n2 && n2>0   % loop for second branch
            while ~f && m2>n2 && n2>=0 && (~downf || ~upf || ~bf) 
                % loop for bidirectional pruning
                if (~upf || ~bf) && n2<ns
                    upprune;            % upwards pruning
                elseif n2>=ns
                    upf=true;
                end
                if ~f &&  m2>=n2 && m2>0 && m2-n2<ms && (~downf || ~bf)
                    downprune;          % downwards pruning
                elseif m2-n2>=ms
                    downf=true;
                end
                bf=true;
            end %pruning loop
            if f || m2<n2 || n2<0
                % pruned cases
                return 
            elseif m2==n2 || ~n2
                % terminal nodes
                break
            end
            if n2==1,
                % case for one more to fix
                q2(~rem)=0;
                [b,idk]=max(q2);
                s=fx;
                s(idk)=true;
                update(s);
                rem(idk)=false;
                m2=m2-1;
                downf=false;
                upf=true;
                continue
            end
            if m2-n2==1,
                % case for one more to remove
                p2(~rem)=0;
                [b,idk]=max(p2);
                rem(idk)=false;
                s=fx|rem;
                update(s);
                fx(idk)=true;
                nf=nf+1;
                n2=n2-1;
                m2=m2-1;
                upf=false;
                downf=true;
                continue
            end
            % save current working data for second branch
            fx1=fx;
            rem1=rem;
            b0=bound;
            p0=p2;
            q0=q2;
            ss=s2;
            X0=Xd;
            dV0=downV;
            dR0=downR;            
            if n2<0.5*m2                %upward branching
                if bound && m2-n2<50
                    p2(~rem)=Inf;
                    [b,id]=min(p2);
                else
                    q2(~rem)=0;
                    [b,id]=max(q2);
                end
                fx(id)=true;
                rem1(id)=false;
                downf=true;
                upf=false;
                bn=bbL2sub(fx,rem1)-1;
                downf=false;
                upf=true;
            else                        % downward branching
                if bound && m2-n2<50
                    p2(~rem)=0;
                    [b,id]=max(p2);
                else
                    q2(~rem)=Inf;
                    [b,id]=min(q2);
                end
                fx1(id)=true;
                downf=false;
                upf=true;
                rem1(id)=false;
                bn=bbL2sub(fx,rem1)-1;
                downf=true;
                upf=false;
                if q0(id)<=bound
                    return
                end
            end
            if bn>0
                bn=bn-sum(rem0)+m2;
                return
            end
            fx=fx1;
            rem=rem1;
            if b0<bound         % if improved bound in the first branch
                bf=false;
                L=q0'<bound;
                if any(L)
                    rem=rem&~L;
                end
            end
            % recover save data before first branch
            f=false;
            p2=p0;
            q2=q0;
            s2=ss;
            nf=sum(fx);
            n2=n-nf;
            m2=sum(rem);
            Xd=X0;
            downV=dV0;
            downR=dR0;
        end
        if ~f   % terminal cases
            if m2==n2
                bn=update(fx|rem)-1;
            elseif ~n2
                bn=update(fx)-1;
            end
        end
        bn=bn+sum(fx0)-nf;
    end

    function downprune
        % downward pruning
        downf=true;
        if ~bound && m2-n2~=1
            return
        end
        s0=fx|rem;
        t=xor(downV,s0);
        if mode && bf && sum(t)==1 && downR(t)
            % single update
            D=Xd(rem,rem);
            x=Xd(rem,t);
            D=D-x*x'/Xd(t,t);
            downV=s0;
            p2(rem)=diag(D);
            Xd(rem,rem)=D;
        elseif bf && isequal(downV,s0) && sum(downR&rem)==m2
            % no pruning
            return;
        else
            % normal cases
            [R1,f]=chol(Y2(s0,s0));
            if f,
                return
            end
            Q=R1\(R1'\eye(m2+nf));
            Yinv(s0,s0)=Q;
            downV=s0;
            V=G(s0,:);
            U=Q*V;
            Gd(s0,:)=U;
            [R,f]=chol(V'*U-eye(n)*bound);
            if f,
                return
            end
            U=R'\Gd(rem,:)';
            if mode
                D=Yinv(rem,rem)-U'*U;
                p2(rem)=diag(D);
                Xd(rem,rem)=D;
            else
                p2(rem)=diag(Yinv(rem,rem))-sum(U.*U,1)';
            end
        end
        ops(2)=ops(2)+1;
        ops(3)=ops(3)+m2;
        downR=rem;
        p2(~rem)=Inf;
        L=p2<=0;
        if any(L)
            % perform downwards pruning
%             disp([m2 n2])
            upf=false;
            fx(L)=true;
            rem(L)=false;
            nf=sum(fx);
            m2=sum(rem);
            n2=n-nf;
        end
    end

    function upprune
        % upwards pruning
        q2(:)=Inf;
        [R1,f]=chol(Y2(fx,fx));
        if f,
            return
        end
        R2=R1'\Y2(fx,rem);
        X1=R1'\G(fx,:);
        [R,f]=chol(X1*X1'-eye(nf)*bound);
        if f,
            return
        end
        R3=diag(Y2(rem,rem)-R2'*R2);
        X2=G(rem,:)-R2'*X1;
        X12=X1*X2';
        X=R'\X12;
        ops(2)=ops(2)+1;
        ops(3)=ops(3)+m2;
        q2(rem)=(sum(X2.*X2,2)-sum(X.*X,1)')./R3;
        upf=true;
        L=q2<=bound;
        if any(L)
            % perform upwards pruning
%             disp([m2 n2])
            downf=false;
            rem(L)=false;
            m2=sum(rem);
            q2(L)=Inf;
        end
    end

    function bf0=update(s)
        % terminal case to update bound
        X=chol(Y2(s,s))'\G(s,:);
        lambda=eig(X'*X);
        ops(1)=ops(1)+1;
        bf0=sum(lambda<bound);
        if ~bf0,
            B(ib)=min(lambda);         %avoid sorting
            sset(ib,:)=find(s);
            [bound,ib]=min(B);
            bf=false;
        end
    end
end
