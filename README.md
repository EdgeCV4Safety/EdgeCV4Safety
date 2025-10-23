# EdgeCV4Safety: AI-Driven Contextual Safety System for Industry 4.0/5.0

## 1. System Overview and Objective 🎯

This project implements a **modular and scalable Computer Vision (CV) system** designed to replace traditional physical barriers, enhancing **worker safety** in industrial settings (Industry 4.0/5.0). The core objective is to achieve **contextual control** of machinery based on the dynamic state of the surrounding work environment.

The system continuously monitors a defined workspace. Upon the detection of personnel entering this area, appropriate countermeasures are instantly triggered, influencing machinery behavior to prevent hazardous situations. This compartmentalized architecture promotes high **modularity and scalability**.

***

## 2. Architectural Components and Design Choices

The architecture is built on flexible components, ensuring low-latency processing critical for safety applications.

### 2.1. Sensing: Industrial Camera

For flexible and high-performance monitoring, the system is configured to use **Industrial Cameras** (no specific brand required).

* **Flexibility & Performance:** The camera can be positioned dynamically within the workspace while providing the necessary data quality for vision processing.
* **Connection:** The camera, for our use-case, was connected via a standard Ethernet to Network, using GigE Vision protocol, to ensure stable and responsive data transmission.

### 2.2. Data Processing: Edge Computing

Effective workspace monitoring requires sophisticated data processing. This computation is thought to be realized on powerful hardware situated **on-site**, leveraging the **Edge Computing** paradigm. Positioning the compute nodes locally minimizes network overhead and guarantees the low-latency required for a real-time safety system.

* **Human Recognition:** A **fast and efficient object detection model (YOLO11)** identifies the presence of personnel.
* **Contextual Awareness:** **High-performance ONNX depth estimation models (UniDepth v2 or DepthAnything v2)** are used to calculate distances, enabling more granular and contextual control responses.
* **Hardware:** The demanding nature of these Deep Learning models requires high-throughput computational nodes equipped with **dedicated Graphical Processing Units (GPUs)**. The advantage is that, thanks to ONNX convertion of AI models, there is no further requested specification. The scripts will provide the best performance with the available hardware.

### 2.3. Control and Action: The Safety Logic Core

Based on the computed results (detection and distance), a core logic module interprets the data and transmits **new directives** to the industrial machinery (e.g., deceleration, emergency stop, or path alteration). This ensures the application of appropriate, data-driven countermeasures.
In the use-case provided, it sends new speed values to a Universal Robots robotic arm, managing to increase or decrease speed, even to stop it.

***

## 3. System Architecture and Flow

The architecture is designed to be geographically close to the point of action (Edge Continuum), providing guaranteed real-time performance. The system's components are interconnected via a standard network, with the compute node(s) acting as the decision-making core.

### 3.1. General Architecture

The overall system architecture is conceptually represented below, illustrating the relationship between sensing, processing, and control elements.

![*General System Architecture.*](assets/general_architecture.png)

For future scalability or in highly distributed environments requiring deterministic timing guarantees, the system is prepared for integration with the **Time-Sensitive Networking (TSN)** protocol.

* **TSN Benefit:** The implementation of TSN (requiring a specialized switch and camera support) would provide **deterministic latency** and prioritize data flow, further strengthening the system's real-time performance for safety-critical applications.

### 3.2. Functional Flow

The functional flow illustrates the complete, contextually driven sequence from data acquisition to the final control action:

![*Functional Flow of the Architecture.*](assets/flow_pipeline.png)


***
