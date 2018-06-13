#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
default_random_engine gen;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  
  //Number of particles
  num_particles = 20;
  weights.resize(num_particles);
  
  // This line creates a normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  
  for(unsigned int i=0; i < num_particles; i++){
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
    weights[i] = 1.0;
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  
  //Normal distributions for sensor noise
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  
  for(unsigned int i=0; i < num_particles; i++){
    if(fabs(yaw_rate) < 0.00001){
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }else{
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - (cos(particles[i].theta + yaw_rate*delta_t)));
      particles[i].theta += yaw_rate * delta_t;
    }
    // add noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
  unsigned int nObs = observations.size();
  unsigned int nPred = predicted.size();
  
  for(unsigned int i = 0; i < nObs; i++) { // For each observation
    double minDist = numeric_limits<double>::max();
    int mapId = -1;
    for(unsigned j = 0; j < nPred; j++ ) { // For each predition.
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      //        cout << "distance[" << i <<"]["<<j<<"]: "<<distance<<endl;
      if(distance < minDist) {
        minDist = distance;
        mapId = predicted[j].id;
        //        cout << "minDist: "<<minDist << endl;
        //        cout << "mapID: "<<mapId << endl;
      }
    }
    //    cout << "mapID: "<<mapId << endl;
    observations[i].id = mapId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  //   You can read more about this distribution here:
  //   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.html
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  
  // STEPS:
  //  1 - For each particle convert measurements to map coordinate
  //  2 - For each observation find the closest landmark
  //  3 - Calculate error for the pair obs x best landmark
  //  4 - Accumulate error for particle
  
  // It stays the same so can be outside the loop
  const double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  
  // The denominators of the mvGd also stay the same
  const double x_denom = 2 * std_landmark[0] * std_landmark[0];
  const double y_denom = 2 * std_landmark[1] * std_landmark[1];
  
  
  //Each particle for loop
  for(int i = 0; i < num_particles; i++){
    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;
    
    
    ////// observations (landmark which car detected in sensor range) /////
    // Define observations in map coordinate
    vector<LandmarkObs> map_obs(observations.size());
    
    // change observations from car coordinate to map coordinate
    for(int j = 0; j < observations.size(); j++){
      LandmarkObs landmark;
      landmark.x = particles[i].x + observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta);
      landmark.y = particles[i].y + observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta);
      landmark.id = observations[j].id;
      map_obs[j] = landmark;
    }
    
    ////// prediction (landmark which located in sensor range from a particle) /////
    vector<LandmarkObs> predictions;
    
    //get landmark position and calculate prediction
    for(int j = 0; j < map_landmarks.landmark_list.size(); j ++){
      double landmark_x = map_landmarks.landmark_list[j].x_f;
      double landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;
      
      // only consider landmarks within the sensor range of the particle
      if(dist(particle_x, particle_y, landmark_x, landmark_y) <= sensor_range){
        predictions.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
      }
    }
    
    // associate predicted landmarks with observed landmarks(sensor measurement)
    dataAssociation(predictions, map_obs);
    
    particles[i].weight = 1.0;
    // find actual x and y coordinates for each observed landmark
    for(int j = 0; j < map_obs.size() ; j++){
      LandmarkObs predicted;
      // find corresponding predicted landmark for current particle
      auto obs = map_obs[j];
      for(int k = 0; k < predictions.size() ; k++){
        if(obs.id == predictions[k].id){
          predicted = predictions[k];
          break;
        }
      }
      //      auto predicted = predictions[obs.id];
      
      auto dx = obs.x - predicted.x;
      auto dy = obs.y - predicted.y;
      
      double exponent = ((dx * dx) / x_denom) + ((dy * dy) / y_denom);
      
      // calculate weight using normalization terms and exponent
      double weight = gauss_norm * exp(-exponent);
      //      if(weight < 0.01){
      //        weight = 0.01;
      //      }
      
      // landmark measurements are independent
      particles[i].weight *= weight;
    }
    weights[i] = particles[i].weight;
//    cout << "i: "<< i << " weight: " << weights[i] << endl;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  discrete_distribution<size_t> dist_index(weights.begin(), weights.end());
  vector<Particle> new_particles;
  
  auto index = dist_index(gen);
  double beta = 0.0;
  double max_weight = *max_element(weights.begin(), weights.end());
  
  uniform_real_distribution<double> distDouble(0.0, max_weight);
  
  for(int i = 0; i < num_particles; i++){
    beta += distDouble(gen) * 2.0;
    while(beta > weights[index]){
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
  
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                         const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the ass   ociations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
