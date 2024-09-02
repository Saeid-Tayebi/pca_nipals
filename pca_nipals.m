function mypca=pca_nipals(data,Num_com,alpha,to_be_scaled)

%%% receives data (in original format), the number of required
%%% components, alpha and return a PCA model including P, T, Rsquared, 
%%% x_hat, t_squared,SPE, tsquared_lim and SPE_lim
        if (nargin<2||isempty(Num_com) || Num_com==0 || Num_com>size(data,2))
            Num_com=size(data,2);
        end
        if nargin<3
            alpha=0.95;
        end
        if nargin<4
            to_be_scaled=1;
        end

        %% pre-processing
        data_origin=data;
        [~,var_rank]=max(var(data_origin));
        Num_obs=size(data,1);
        Cx=mean(data);
        Sx=std(data)+1e-16;

         if to_be_scaled
            data=(data-Cx)./Sx;
         end
        X=data;
        P=zeros(size(X,2),Num_com);
        T=zeros(size(X,1),Num_com);
        SPE=zeros(size(T));
        Rsquare=zeros(1,Num_com);
        SPE_lim=zeros(1,Num_com); 
        tsquared=zeros(Num_obs,Num_com);
        T2_lim=zeros(1,Num_com);
        ellipse_radius=zeros(1,Num_com);
        covered_var=zeros(1,Num_com);


        
        %% Nipals Algorithem
        
        for i=1:Num_com
        
            b=var_rank;
            t1=X(:,b);
            
            while true
     
                P1=(t1'*X)/(t1'*t1);
                P1=P1./norm(P1);
                tnew=((P1*X')/(P1*P1'))'; 

                told=t1;
                t1=tnew;
                
                E=tnew -told;
                E=E'*E;
                if E<1e-15
                    break
                end
            end
            
            xhat=t1*P1;
            Enew=X-xhat;
            X=Enew;
         
            P(:,i)=P1;
            T(:,i)=t1;
            covered_var(i)=var(t1);

            % SPE
            [SPE(:,i),SPE_lim(i),Rsquare(i)]=SPE_calculation(T, P,data,alpha);

            %T2
            [tsquared(:,i), T2_lim(i),ellipse_radius(i)]=T2_calculations(T(:,1:i),i,Num_obs,alpha);
        
        end

       
%% Function output
        mypca.P=P;
        mypca.T=T;
        mypca.Rsquare=Rsquare;
        mypca.covered_var=covered_var;
        mypca.X_hat=T*P';
        mypca.tsquared=tsquared;
        mypca.T2_lim=T2_lim;
        mypca.ellipse_radius=ellipse_radius;
        mypca.SPE_x=SPE;
        mypca.SPE_lim_x=SPE_lim;
        mypca.x_scaling=[Cx;Sx];
        mypca.Xtrain_normal=data_origin;
        mypca.Xtrain_scaled=data;
        mypca.alpha=alpha;
end

function [spe,spe_lim,Rsquare]=SPE_calculation(score, loading,Original_block,alpha)
%%% receive score,loading, original block (scaled format) and alpha, and calculate the Error
%%% and SPE and the SPE_lim as well as Rsquared

            X_hat=score*loading';
            Error=Original_block-X_hat;
            spe=sum(Error.*Error,2);
            m=mean(spe);
            v=var(spe);
            spe_lim=v/(2*m)*chi2inv(alpha,2*m^2/v);

            %Rsquared
            Rsquare=(1-var(Error)/var(Original_block));
             
end

function [tsquared, T2_lim,ellipse_radius]=T2_calculations(T,Num_com,Num_obs,alpha)
%%% recieve Score Matrix, the current applied number of components,num of
%%% observations and alpha and return all Hotelling T2 related calculations
%%% including tsquared, T2_lim and ellipse_radius
            tsquared=sum((T./std(T)).^2,2);
            T2_lim=(Num_com*(Num_obs^2-1))/(Num_obs*(Num_obs-Num_com))*finv(alpha,Num_com,(Num_obs-Num_com));
            ellipse_radius=(T2_lim*std(T(:,Num_com))^2)^0.5;

end

