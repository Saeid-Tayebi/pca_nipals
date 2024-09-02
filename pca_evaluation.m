function [x_hat,t_point,SPE,tsquared,x_new_scaled]=pca_evaluation(pca_model,x_new)
    %%% receive a pca model and new observation (just X or Y) in un scaled
    %%% format and calculate t score and spe and HotelingT2 for that

    Cx=pca_model.x_scaling(1,:);
    Sx=pca_model.x_scaling(2,:);

    x_new_scaled=(x_new-Cx)./Sx;                 %scaled
    t_point=x_new_scaled*pca_model.P;       %t_scor of the new point
    x_hat=t_point*pca_model.P';                          % scaled

    x_hat=(x_hat.*Sx)+Cx;

    [SPE,~,~]=SPE_calculation(t_point,pca_model.P,x_new_scaled,pca_model.alpha);

    tsquared=sum((t_point./std(pca_model.T)).^2,2);

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