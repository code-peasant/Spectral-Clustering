function [ Jac_Match_Plain , Jac_Match_Slow ,  Jac_Match_Fast ] ...
                                            = gen_data_me( n,k,pNoise )
clc; close all;

%rand('state', 12);   % comment out if you want a random problem each time
% this just set the 'seeds' to the random number generator, so that you 
% get the same result each time you run the code.

% n = 50;
% k = 5;
% pNoise = 0.45;

N = n*k;

G = sparse(N,N);
Network = struct([]);

useOrderedLabels = 1;
if useOrderedLabels == 1;
    ctr = 0;
    for i = 1:k
        Network(i).labels =  (ctr+1) : (ctr+n);
        ctr = ctr + n;
    end
else
    someRandomPerm = randperm(N);
    ctr = 0;
    for i = 1:k
        Network(i).labels =  someRandomPerm( (ctr+1) : (ctr+n) );
        ctr = ctr + n;
    end
end


for i = 1 : (k-1)
    for j = (i+1) : k
        G(  Network(i).labels ,  Network(j).labels ) = eye(n);
    end
end
G = G + G';

% Build the ground truth "clusters"  (sets of matching)
% Example: (when we use the nice consecutive indices/labeling)
%       Cluster(1).Labels = {1, 51, 101,151}
%       Cluster(2).Labels = {2, 52, 102, 152}
%       ...
%       Cluster(n).Labels = {50, 100, 150, 200}
STACKED_LABELS = [];
for i = 1 : k
   STACKED_LABELS = [ STACKED_LABELS;  Network(i).labels ];
end
disp(STACKED_LABELS);

% ALL_LABELS: column 1 is cluster 1, .... , column n is cluster n
OriginalClusters = struct([]);
for i = 1 : n
	OriginalClusters(i).Labels = STACKED_LABELS(:,i);
    % G( OriginalClusters(i).Labels , OriginalClusters(i).Labels ) should
    % be all ones.
end


subplot(2,3,1);
spy(G); title('Clean Graph');
% Add noise. With probability p add a new edge, or delete an existing one

NoiseGraph = sparse(ErdosRenyi(N,pNoise));
%%% make sure the noise matrix is symmetric!
NoiseGraph = triu(NoiseGraph,1);  NoiseGraph = NoiseGraph + NoiseGraph';
% Zero-out the diagonal blocks of the noise graph
for i = 1:k
   NoiseGraph( Network(i).labels, Network(i).labels) = 0;
end

subplot(2,3,2);
spy(NoiseGraph);  title('Pure Noise Graph');

H = mod(G + NoiseGraph,2);
H - H'
subplot(2,3,3);
spy(H);  title('Noisy Graph');


%% Whether to run the local matchings between pairs of networks
runLocaMatching = 1;

if runLocaMatching == 1
    %%%% Compute the local pairwise matchings between networks:
    Edges_MG = [];
    for i = 1:(k-1)
        for j = (i+1) : k
            % Compute a local max matching
            SIM = H(  Network(i).labels ,  Network(j).labels);
            iNbrs = Network(i).labels;
            jNbrs = Network(j).labels;
            [ MatchingVal, MATCHING ] = compBipMatching(SIM, iNbrs, jNbrs);
            % MatchingVal;  % uncomment to the the format of the output..
            % MATCHING;
            
            % Put the matching back:
            Edges_MG = [ Edges_MG;  MATCHING ];
            % input('seee');
        end
    end
    size(Edges_MG);
    
    % Create a new graph from the above matchings
    MG = sparse( Edges_MG(:,1), Edges_MG(:,2), Edges_MG(:,3), N, N); % the "match-graph";
    MG = MG + MG';
    
    subplot(2,3,4);
    spy(MG);  title('Denoised graph MG');
    
    dif_G_MG = abs(G - MG);
    nrMismatches = sum(sum(dif_G_MG))/2  % missing or extra edges with respect to G
    disp( ['Number of mismatches between G and MG =' int2str(nrMismatches)] );
end



nrClust = n;
alpha = 5;
PARS.runPlainClust  = 1; % whether too run plain (un-constrained) clustering
PARS.runGenEigs     = 1; % whether too run MATLAB's generalized eig value problem (SLOW)
PARS.runGenPowFast  = 1; % whether too run the FAST method 

Q = sparse( N,N );          %%%%  %  (ML = -1).  (CL = +1);
for i = 1 : k
    Q( Network(i).labels, Network(i).labels) = 1;
end
Q = Q - diag(diag(Q));   % set the diagonal of Q to 0

addpath(genpath('/Users/Mihai/Google Drive/MATLAB/ConClust_Clean'));
%%%% [ CLEAN_STATS ] = wrapper_consClust(H + 2* MG, Q, nrClust, alpha, PARS );
% [ CLEAN_STATS ] = wrapper_consClust(MG, Q, nrClust, alpha, PARS );
[ CLEAN_STATS ] = wrapper_consClust(H, Q, nrClust, alpha, PARS );
CLEAN_STATS;

%   IDX_PLAIN    IDX_SLOW    IDX_FAST

showEmb = 0;
if showEmb == 1
    disp('CLEAN_STATS.EMB_PLAIN::::');  disp(CLEAN_STATS.EMB_PLAIN);
    input('next see EMB_PLAIN...');
    
    disp('CLEAN_STATS.EMB_SLOW1::::');  disp(CLEAN_STATS.EMB_SLOW1);
    disp('CLEAN_STATS.EMB_SLOW2:::');   disp(CLEAN_STATS.EMB_SLOW2);
    input('next see SLOW...');
    
    disp('CLEAN_STATS.EMB_FAST1::::');  disp(CLEAN_STATS.EMB_FAST1);
    disp('CLEAN_STATS.EMB_FAST2:::');   disp(CLEAN_STATS.EMB_FAST2);
    input('next see FAST...');
    
    disp(CLEAN_STATS.IDX_ALL);
    input('next see IDX_ALL...');
end

 %[ RecoveredClusters , vectorSizes] = getClusters(n, CLEAN_STATS.IDX_PLAIN );

% avgJaccag_Matching = compute_Quality(OriginalClusters , CLEAN_STATS.IDX_PLAIN, n)
% avgJaccag_Matching = compute_Quality(OriginalClusters , CLEAN_STATS.IDX_SLOW, n)
% avgJaccag_Matching = compute_Quality(OriginalClusters , CLEAN_STATS.IDX_FAST, n)


%% Compute clustering

[IDX_PLAIN,IDX_SLOW,IDX_FAST ] = ...
    allMethods_Clustering_Locally(CLEAN_STATS, PARS, Network, nrClust, k);

Jac_Match_Plain = NaN;    Jac_Match_Slow= NaN;    Jac_Match_Fast = NaN;

Jac_Match_Plain = compute_Quality(OriginalClusters , IDX_PLAIN, n, H);%%%%%%%%%%%%%%%%%%
Jac_Match_Slow = compute_Quality(OriginalClusters , IDX_SLOW, n, H);
Jac_Match_Fast = compute_Quality(OriginalClusters , IDX_FAST, n, H);


% In H, for each pair (ij) of networks (stored in Network)
% consider the bipartite networks with bipartitions T_i and T_j
% (this would just be a submatrix of H, denote this submatrix by S_{ij})

% (using the attached code maxWeightMatching.m): 
% Compute a maximum weight matching in S_{ij} and then  set to zero all the
% entries in S_{ij}) and place 1's for the entries corresponding to the
% matching. Then "put back" into H (or, better, build a new matrix F) the
% submatrix S_{ij}.

% Repeat the above steps for all pairs of networks i<j, and "fill" in F one
% step at a time.

% Perform spectral clustering in F. Even better, perform constrained
% spectral clustering in F, where you add some constraints (since we de not
% want to have two labels of the same network ending up in the same
% cluster). For this step we can use my readily availbale algorithm, see
% Section 11 in the projects PDF file. Just a one line call on your end. 

end


function [ RecoveredClusters , vectorSizes] = getClusters(n, IDX)

vectorSizes = [];
for i = 1 : n
    RecoveredClusters(i).Labels = find( IDX == i );
    % disp(RecoveredClusters(i).Labels);  input('see');
    RecoveredClusters(i).Size = length(RecoveredClusters(i).Labels);
    vectorSizes = [ vectorSizes RecoveredClusters(i).Size ];
end
sort(vectorSizes);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [RecoveredClusters] = rebalance(RecoveredClusters,H)
% This function will rebalance the clusterings in RecoveredClusters
greaterClustersIndex = double.empty(1,0);
smallerClustersIndex = double.empty(1,0);
% put the indices of clusters with different sizes in different groups
for i = 1:50
    if RecoveredClusters(i).Size > 5
        greaterClustersIndex = [greaterClustersIndex i];
    end
    if RecoveredClusters(i).Size < 5
        smallerClustersIndex = [smallerClustersIndex i];
    end
end
if (isempty(greaterClustersIndex) == 1) && (isempty(smallerClustersIndex) == 1)
        return;
    end
while isempty(greaterClustersIndex) ~= 1
    index = greaterClustersIndex(1,1);
    Vertex = 0;
    MaxCutedge = 0;
    ClusterIndex = 0;
    for i = smallerClustersIndex % loop through every clustering with size smaller than 5
       for j = (RecoveredClusters(index).Labels)'%(:,1) % loop trhough every vertex in the first bigger clustering
           tempCluster = [RecoveredClusters(i).Labels;j]; % add this element to the end of a clustering
           tempMaxCutedge = countCutedge(tempCluster,RecoveredClusters(index).Labels, H);
           if tempMaxCutedge > MaxCutedge
               MaxCutedge = tempMaxCutedge;
               Vertex = j;
               ClusterIndex = i;
           end
       end
    end
    RecoveredClusters(ClusterIndex).Labels = [RecoveredClusters(ClusterIndex).Labels;j]; % add this element to the new clustering
    RecoveredClusters(ClusterIndex).Size = 1 + RecoveredClusters(ClusterIndex).Size; % change the size
    temp = RecoveredClusters(index).Labels ~= j; % unneccessary, just for clearness
    RecoveredClusters(index).Labels= RecoveredClusters(index).Labels(temp); % delete this element from the old clustering
    RecoveredClusters(index).Size = RecoveredClusters(index).Size - 1; % change the size
    % remove all groups with size of 5
    if isempty(greaterClustersIndex) == 1
        return;
    end
    temp = size(greaterClustersIndex,2);
    i = 1;
    while i<= temp
        if RecoveredClusters(greaterClustersIndex(1,i)).Size == 5
            greaterClustersIndex(i) = [];
            temp = temp - 1;
        end
        i = i+1;
    end
    if isempty(smallerClustersIndex) == 1
        return;
    end
    temp = size(smallerClustersIndex,2);
    i = 1;
    while i<= temp
        if RecoveredClusters(smallerClustersIndex(1,i)).Size == 5
            smallerClustersIndex(i) = [];
            temp = temp - 1;
        end
        i = i+1;
    end
end
end

function [amount] = countCutedge(clustering1, clustering2, SIM)
amount = 0;
for i = 1:size(clustering1,1)
    for j = i:size(clustering2,1)
        if SIM(clustering1(i,1),clustering2(j,1)) ~= 0
            amount = amount + 1;
        end
    end
end
end

function [] = ClusterIndexDelete(ClusterIndex, RecoveredClusters)
if isempty(ClusterIndex) == 1
    return;
end
temp = size(ClusterIndex,2);
i = 1;
while i<= temp
    if RecoveredClusters(ClusterIndex(1,i)).Size == 5
        ClusterIndex(i) = [];
        temp = temp - 1;
    end
    i = i+1;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function JI = JacIndex(A,B)
    JI = length(intersect(A,B)) / length(union(A,B));
end




function [IDX_PLAIN,IDX_SLOW,IDX_FAST ] = ...
    allMethods_Clustering_Locally(CLEAN_STATS,PARS, Network, nrClust, k)

IDX_PLAIN = [];     IDX_SLOW = [];      IDX_FAST = [];

if PARS.runPlainClust  == 1
    IDX_PLAIN = clusterLocally( CLEAN_STATS.EMB_PLAIN, Network, nrClust, k);
end

if PARS.runGenEigs == 1
    IDX_SLOW = clusterLocally( CLEAN_STATS.EMB_SLOW2, Network, nrClust, k);
end

if PARS.runGenPowFast  == 1
    IDX_FAST = clusterLocally( CLEAN_STATS.EMB_FAST2, Network, nrClust, k);
end



end


function IDX = clusterLocally( EMB, Network, nrClust, k)

try 
TD_SEEDS = zeros(nrClust,nrClust,k);
for i = 1:k
    seed_labels = Network(i).labels;
    TD_SEEDS(:,:,i) = EMB(seed_labels,:);
end

IDX = kmeans(EMB, nrClust, 'Start', TD_SEEDS);

catch me
    disp('Error. Will skip.');
    IDX = [];
end

end


function avgJaccag_Matching = compute_Quality(OriginalClusters, IDX, n, H)

try 
[ RecoveredClusters , vectorSizes] = getClusters(n, IDX);
RecoveredClusters = rebalance(RecoveredClusters, H);
sort(vectorSizes);

%% compute the jaccard similarity index between the n initial clusters and
% the recovered n clusters, and find a maximum weight matching in there
% Just so we know whom to whom we should match... and compare results.

JAC = zeros(n,n);
for i = 1 : n
    for j = 1 : n
        JAC(i,j) = JacIndex(OriginalClusters(i).Labels,RecoveredClusters(j).Labels);
    end
end
JAC;
subplot(2,3,[5 6]);  imagesc(JAC); colorbar;  title('jaccard sim matrix'); axis square;
xlabel('recovered clusters');  ylabel('original clusters');  

[ MatchingVal, MATCHING ] = compBipMatching(JAC, 1:n, 1:n);

MatchingVal;
MATCHING;

avgJaccag_Matching = sum( MATCHING(:,3) ) / n;

catch ME
    disp('Error. Will skip this case.');
    avgJaccag_Matching = NaN;
end

end
