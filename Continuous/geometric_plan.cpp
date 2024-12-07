#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/SE2StateSpace.h>

  
#include <ompl/config.h>
#include <iostream>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/base/terminationconditions/CostConvergenceTerminationCondition.h>
// #include <ompl/geometric/planners/prm/PRMstar.h>
#include <ompl/tools/benchmark/Benchmark.h>
#include <ompl/tools/multiplan/ParallelPlan.h>

#include <cmath>
#include <vector>
#include <sstream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>

namespace ob = ompl::base;
namespace og = ompl::geometric;
namespace ot = ompl::tools;

std::vector<std::string> scenario_map;
double scenario_step = 0;


ob::PlannerPtr createPlanner(const ob::SpaceInformationPtr &si)
{
    return std::make_shared<og::RRTstar>(si);
}



bool isValid(const ob::State *state) {
    // cast the abstract state type to the type we expect
    const auto *se2state = state->as<ob::SE2StateSpace::StateType>();
    // colisão
    const auto *pos = se2state->as<ob::RealVectorStateSpace::StateType>(0);

    const double x = (*pos)[0];  // Access the x-coordinate
    const double y = (*pos)[1];  // Access the y-coordinate

    //if (!(scenario_map[static_cast<int>(round(x / scenario_step))][static_cast<int>(round(y / scenario_step))] == '.')){
    //    return false;
    //}
    
    for (int step = 0; step < 4; ++step) {
        double radius = scenario_step * step;
        for (int a = 0; a < 8; ++a) {
            double x_radius = round((cos(45 * a * M_PI / 180) * radius + x) / scenario_step);
            double y_radius = round((sin(45 * a * M_PI / 180) * radius + y) / scenario_step);

            if (x_radius > 511) {
                x_radius = 511;
            }
            if (x_radius < 0) {
                x_radius = 0;
            }

            if (y_radius > 511) {
                y_radius = 511;
            }
            if (y_radius < 0) {
                y_radius = 0;
            }

            if (!(scenario_map[static_cast<int>(x_radius)][static_cast<int>(y_radius)] == '.') ||
                x_radius == 511 || y_radius == 511 || x_radius == 0 || y_radius == 0) {
                return false;
            }
            if (step == 0){
                break;
            }
        }
    }

    return true;
}

  
std::string plan(const std::vector<double>& start, const std::vector<double>& goal, const std::vector<std::string>& scenario_map1, const double scenario_step1, const int num_planners, const int seed)
{
    scenario_map = scenario_map1;
    scenario_step = scenario_step1;

    std::uint_fast32_t convertedSeed = static_cast<std::uint_fast32_t>(seed + 1);
    ompl::RNG::setSeed(convertedSeed); 
     
     // construct the state space we are planning in
    auto space(std::make_shared<ob::SE2StateSpace>());

     // construct an instance of  space information from this state space
    ob::SpaceInformationPtr si(new ob::SpaceInformation(space));

    auto pdef(std::make_shared<ob::ProblemDefinition>(si));
  
    // set the bounds for the R^2 part of SE(2)
    ob::RealVectorBounds bounds(2);
    bounds.setLow(0);
    bounds.setHigh(10);
  
    space->setBounds(bounds);


    // Set the state validity checker
    si->setStateValidityChecker(isValid);
  
    // create a random start state
    ob::ScopedState<> startState(space);
    startState[0] = start[0];
    startState[1] = start[1];
  
    // create a random goal state
    ob::ScopedState<> goalState(space);
    goalState[0] = goal[0];
    goalState[1] = goal[1];

    pdef->setStartAndGoalStates(startState, goalState);

    // ot::ParallelPlanPtr pp(new ot::ParallelPlan(pdef));


    // Create multiple planner instances
    // for (int i = 0; i < num_planners; ++i)
    // {
    //    ob::PlannerPtr planner = createPlanner(si);
    //   // planner->as<og::RRTstar>()->setRange(1);
    //    planner->setProblemDefinition(pdef);
    //    pp->addPlanner(planner);     
    // }

    ob::PlannerPtr planner = createPlanner(si);
    planner->setProblemDefinition(pdef);

    ob::PlannerTerminationCondition ter1 = ob::CostConvergenceTerminationCondition(pdef, 20, 0.01);
    ob::PlannerTerminationCondition ter2 = ob::timedPlannerTerminationCondition(20);
    ob::PlannerTerminationCondition cond = ob::plannerOrTerminationCondition(ter1, ter2);

    ob::PlannerStatus solved = planner->solve(cond);
    
    if (solved)
    {
        // get the goal representation from the problem definition (not the same as the goal state)
        std::vector<ob::PlannerSolution> paths = pdef->getSolutions();

        // Iterate through the vector and print each solution
        std::stringstream output;
        for (const auto& solution : paths) {
            ob::PathPtr path = solution.path_;

            // Check if the path is not null before printing
            if (path) {
                // std::cout << "Found solution:" << std::endl;
                path->as<og::PathGeometric>()->printAsMatrix(output);
            }
        }

        // Return the accumulated output as a string
        return output.str();
    }
    else {
        return "Path is null.";
        }   
}

// Binding code
PYBIND11_MODULE(geometric_plan_c, m) {
    m.def("plan", &plan, "Plan function");
}
