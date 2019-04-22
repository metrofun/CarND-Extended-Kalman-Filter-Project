#include "tools.h"
#include <iostream>
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

const float Tools::ZERO_EPS = 1e-5;
bool Tools::IsZero(float value)
{
   return fabs(value) < Tools::ZERO_EPS;
}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
   VectorXd rmse(4);
   rmse << 0, 0, 0, 0;

   // check the validity of the following inputs:
   //  * the estimation vector size should not be zero
   //  * the estimation vector size should equal ground truth vector size
   if (estimations.size() != ground_truth.size() || estimations.size() == 0)
   {
      std::cout << "Invalid estimation or ground_truth data" << std::endl;
      return rmse;
   }

   // accumulate squared residuals
   for (unsigned int i = 0; i < estimations.size(); ++i)
   {

      VectorXd residual = estimations[i] - ground_truth[i];

      // coefficient-wise multiplication
      residual = residual.array() * residual.array();
      rmse += residual;
   }

   // calculate the mean
   rmse = rmse / estimations.size();

   // calculate the squared root
   rmse = rmse.array().sqrt();

   // return the result
   return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state)
{

   MatrixXd Hj(3, 4);
   // recover state parameters
   float px = x_state(0);
   float py = x_state(1);
   float vx = x_state(2);
   float vy = x_state(3);

   // pre-compute a set of terms to avoid repeated calculation
   float c1 = px * px + py * py;
   float c2 = sqrt(c1);
   float c3 = (c1 * c2);

   // check division by zero
   if (Tools::IsZero(c1))
   {
      std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
      return Hj;
   }

   // compute the Jacobian matrix
   Hj << (px / c2), (py / c2), 0, 0,
       -(py / c1), (px / c1), 0, 0,
       py * (vx * py - vy * px) / c3, px * (px * vy - py * vx) / c3, px / c2, py / c2;

   return Hj;
}

Eigen::VectorXd Tools::CalculateRadarSpace(const Eigen::VectorXd &x_state)
{
   VectorXd z(3);
   // recover state parameters
   float px = x_state(0);
   float py = x_state(1);
   float vx = x_state(2);
   float vy = x_state(3);

   float rho = hypot(px, py);
   float rho_dot, phi;

   // check division by zero
   if (fabs(rho) < Tools::ZERO_EPS)
   {
      std::cout << "CalculateRadarSpace() - Error - Division by Zero" << std::endl;
      rho_dot = 0;
   } else {
      rho_dot = (px * vx + py * vy) / rho;
   }

   if (Tools::IsZero(py) && Tools::IsZero(px)) {
      std::cout << "CalculateRadarSpace() - Error - Atan2 Domain Error" << std::endl;
      phi = 0;
   } else {
      phi = atan2(py, px);
   }

   z << rho, phi, rho_dot;
   return z;
}