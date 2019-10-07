% Variables for this are {0,1}!!!
function energy = hamilGen(ffunc, state, coupls)

nN = numel(state);
nF = numel(coupls);
if(size(state,2)~= nN); state = state'; end
if(size(state,2)~= nF); coupls = coupls'; end

energy = -[state feval(ffunc,state')'] * coupls';
