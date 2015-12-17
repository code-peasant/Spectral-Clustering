function [ output_args ] = run_many( input_args )
close all;

n = 50;
k = 5;

ALL_JAC = [];
label_experiment = [ 'Results_n' int2str(n) ' _k' int2str(k) ];
myFileName = [ label_experiment '.mat' ];

doWork = 1;  %% set to 0 if you did the work before, and want to reload results
if doWork == 1
    
    pNoiseVector = 0 :0.05: 0.5;

    for pNoise = pNoiseVector
        [ Jac_Match_Plain , Jac_Match_Slow ,  Jac_Match_Fast ] = ...
            gen_data_me(n, k, pNoise);
        ALL_JAC = [ ALL_JAC;   Jac_Match_Plain  Jac_Match_Slow   Jac_Match_Fast ]
    end
    
    save(myFileName, 'ALL_JAC','pNoiseVector');  disp('Saved all results');
else
    load(myFileName, 'ALL_JAC','pNoiseVector'); disp('Loaded all results');
end
pNoiseVector



%% plot results, and save figure to file:
close all;
figure(1)
plot( pNoiseVector, ALL_JAC(:,1), '-ro', 'LineWidth', 3, 'MarkerSize',15); hold on
plot( pNoiseVector, ALL_JAC(:,2), '-b*', 'LineWidth', 3, 'MarkerSize',15); hold on
plot( pNoiseVector, ALL_JAC(:,3), '-gs', 'LineWidth', 3, 'MarkerSize',15); % axis tight;
set(gca,'FontSize',18);
title(  regexprep(label_experiment, '_', '  '), 'Fontsize',18);

xlabel('Noise level');          ylabel('Average Jaccard Index');
h_legend = legend('Plain','Constrained-SLOW','Constrained-FAST', 'Location','best' );
set(h_legend,'FontSize',18);

% saveas(1,label_experiment,'epsc'); 
saveas(1,label_experiment,'fig');   saveas(1,label_experiment,'png');

end

