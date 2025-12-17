import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Physical AI Fundamentals',
    Svg: null, // We'll use a placeholder since we don't have actual SVGs
    description: (
      <>
        Learn the foundational concepts of Physical AI, where artificial intelligence
        meets physical systems through embodied cognition and real-world interaction.
      </>
    ),
  },
  {
    title: 'Humanoid Robotics',
    Svg: null,
    description: (
      <>
        Master the design and control of humanoid robots with advanced locomotion,
        manipulation, and human-robot interaction capabilities.
      </>
    ),
  },
  {
    title: 'Simulation to Reality',
    Svg: null,
    description: (
      <>
        Bridge the reality gap with advanced sim-to-real transfer techniques
        using NVIDIA Isaac Sim and Isaac ROS frameworks.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className={clsx('card', styles.featureCard)}>
        <div className="card__body text--center">
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}