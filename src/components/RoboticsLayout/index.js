import React from 'react';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

export default function RoboticsLayout(props) {
  const {siteConfig} = useDocusaurusContext();
  const {children, title, description} = props;

  return (
    <Layout
      title={title}
      description={description}
      wrapperClassName="robotics-layout-wrapper">
      <div className="robotics-content-container">
        {children}
      </div>
    </Layout>
  );
}