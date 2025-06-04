# Datasets

[//]: HTTPS

<details> 
  <summary> legend </summary> 

- Weak: Train: Video-Level // Test: Frame-Level
- TS: TimeStamps (transforming to FL is possible, so basicaly TS=FL)

</details> 

<!-- https://ryozomasukawa.github.io/PV-VTT.github.io/# -->
<table><thead>
  <tr>
    <th rowspan="2">Year</th>
    <th rowspan="2">Dataset <br></th>
    <th rowspan="2">Details<br></th>
    <th colspan="2" style="text-align: center">Video</th>
    <th rowspan="2"># Anomaly</th>
    <th colspan="7" style="text-align: center">Annotations</th>
  </tr>
  <tr>
    <th>#V/Th</th>
    <th>Audio</th>
    <th></th>
    <th>Temp</th>
    <th>QA</th>
    <th>Desc</th>
    <th>Judge</th>
    <th>Reas</th>
    <th>CoT</th>
  </tr></thead>
<tbody>
  <tr>
    <td>2018</td>
    <td>UCF-Crime <a href="https://www.crcv.ucf.edu/projects/real-world/">:link:</a> <a href="https://github.com/WaqasSultani/AnomalyDetectionCVPR2018">OG:file_folder:</a> <a href="https://github.com/Roc-Ng/DeepMIL">Torch:file_folder:</a> <a href="https://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf">:newspaper:</a></td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/" width="1500"/>
      </details>
    </td> 
    <td>1900/128h</td>
    <td>-</td>
    <td>13</td>
    <td></td> 
    <td>Weak</td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
  </tr>

  <tr>
    <td>2020</td>
    <td>
      XD-Violence <a href="https://roc-ng.github.io/XD-Violence/">:link:</a> <a href="https://github.com/Roc-Ng/XDVioDet">:file_folder:</a>
    </td> 
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/" width="1500"/>
      </details>
    </td> 
    <td>4754/217h</td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td>6</td>
    <td></td> 
    <td>Weak</td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>  
  </tr>

  <tr>
    <td>2024</td>
    <td>Glance VAD <a href="HTTPS">:file_folder:</a> 
    </td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/" width="1500"/>
      </details>
    </td> 
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/" width="1500"/>
      </details>
    </td>
    <td>Glance</td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
  </tr>

  <tr>
    <td>2024</td>
    <td>MSAD <a href="https://msad-dataset.github.io/">:link:</a> </td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/2024-MSAD.png" width="1500"/>
      </details>
    </td> 
    <td>720/4h</td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td></td>
    <td></td> 
    <td>Weak</td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
  </tr>

  <tr>
    <td>2025</td>
    <td>VADD</td>
    <td>
      <details> 
      <summary>  </summary>
      Video Anomaly Detection Dataset is as an extension of UCF-Crime with 2,591
      videos (2,202 train, 389 test) spanning 18 classes, including underrepresented anomalies
      like road accidents and dangerous throwing from UCFC, Throwing Action [119], and newly
      collected accident videos annotated, with video-level labels (train) and frame-level anomaly
      timestamps (test).
      </details>
    </td> 
    <td>2591/-</td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td>18</td>
    <td></td> 
    <td>Weak</td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
  </tr>

  <tr>
    <td>2025</td>
    <td>UCFC-DVS <a href="HTTPS">:file_folder:</a> </td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/2025-UCFDVS.png" width="1500"/>
      </details>
    </td> 
    <td>-</td>
    <td>-</td>
    <td>13</td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/2025-UCFDVS-anot.png" width="1500"/>
      </details>
    </td>
    <td>Event</td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
  </tr>

  <tr>
    <td>2025</td>
    <td>
      UCFC-HN <a href="HTTPS">:file_folder:</a> </br>
      MSAD-HN
    </td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/2025-HN.png" width="1500"/>
      </details>
      <details> 
        <summary>  </summary> 
        - 1. Goal: Evaluate model overfitting by generating normal videos with the same scene as abnormal videos. </br>
        - 2. Steps for Dataset Construction :</br>
            Step 1: Select one keyframe from Abnormal Train videos of UCF-Crime and MSAD for diffusion models to generate high-quality videos. </br>
            Step 2: Generate Videos: 
              <!-- Models Used : UCF-HN -> Vidu 2.0 (Vidu platform) with resolution 1088×800 at 32 fps / MSAD-HN -> wan2.1_i2v_480p_14B (ComfyUI) with resolution 832×480 at 16 fps.</br>     -->
              (1) Generate a clip from the selected keyframe; (2) Extend the clip using the last frame of the first clip; (3) Temporally concatenate clips. /// UCF-HN: 15-second sequences. / MSAD-HN: 10-second sequences. /// Manual prompts ensure diverse event representations in generated videos. </br>
            Step 3: Manually inspect all retained videos for visual fidelity -> Final datasets:  UCF-HN : 100 videos. / MSAD-HN : 67 videos </br>
          - 3. Validation of Realism : </br> Visual Comparison (Figure 8(a)): Original anomaly video vs. generated normal counterpart.Violent actions replaced by natural walking, maintaining background coherence. </br>
            Feature-Space Analysis (Figure 8(b)): Hard negative samples (generated normals) are closer to abnormal videos than original normal videos. //// This indicates higher realism and similarity to abnormal scenes. </br>
          - 4. Benchmarking: Train models on UCF-Crime or MSAD. Test on corresponding hard normal set (UCF-HN or MSAD-HN). Since all test videos are normal, false alarm rate (FAR)  reflects overfitting degree.
          <!-- 6. Limitations  
            Computational efficiency constraints limit the number of generated videos.
            Complexity of abnormal actions restricts generation of multi-shot videos.
            Excludes XD-Violence due to inability to generate such scenarios. -->
      </details>      
    </td> 
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>
      <details> 
        <summary>  </summary> <img src="img/ds/2025-HN-anotucfc.png" width="1500"/>
      </details> 
      <details> 
        <summary>  </summary> <img src="img/ds/2025-HN-anotmsad.png" width="1500"/>
      </details>
    </td>
    <td>-</td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
  </tr>  


  <tr>
    <td style="text-align: center; font-weight:bold" colspan="13">VAU</td>
  </tr>


  <tr>
    <td>2024</td>
    <td>HAWK <a href="HTTPS">:file_folder:</a> </td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/2024-HAWK.png" width="1500"/>
      </details>
    </td> 
    <td>7898/142.5h</td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td>-</td>
    <td>
      <!-- <details> 
      <summary>  </summary> <img src="img/ds/" width="1500"/>
      </details> -->
    </td>
    <td>-</td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
  </tr>

  <tr>
    <td>2024</td>
    <td>UCFA <a href="HTTPS">:file_folder:</a> </td>
    <td>
      <details> 
      <summary>  </summary> 
      UCFA extends UCF-Crime as the first large-scale multimodal dataset for
      Temporal-Specific Video Generation (TSGV), Video Captioning (VC), Dynamic Video Caption-
      ing (DVC), and Multimodal Anomaly Detection (MAD) tasks. </br> 
      Annotated with over 23,000 sentence-level descriptions and 0.1s-precision
      timestamps. It provides 110.7 hours of data with concise, event-specific language. </br>
      UCFA emphasizes fine-grained temporal grounding and domain relevance, supporting advanced tasks like captioning and query-based retrieval in dynamic, low-quality surveillance contexts.</br>
      Dataset splits and ethical protocols are included to promote reproducible research. Ethical considerations and dataset splits (train/val/test)
      are detailed to support reproducible, responsible research in this emerging field.
      </details>
    </td> 
    <td>1854/122h</td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td>13</td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/2024-UCFA-anot.png" width="1500"/>
      </details>
    </td>
    <td>TS</td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
  </tr>

  <tr>
    <td>2024</td>
    <td>HIVAU-70k <a href="HTTPS">:link:</a> <a href="HTTPS">:file_folder:</a> 
    <td>      
      <details> 
      <summary>  </summary> <img src="img/ds/2025-HIVAU70K0.png" width="1500"/>
      </details>
    </td> 
    <td>5443/-</td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td></td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/2025-HIVAU70K-anotfreetext.png" width="1500"/>
      </details>
      <details> 
      <summary>  </summary> <img src="img/ds/2025-HIVAU70K-anotcl.png" width="1500"/>
      </details>
      <details> 
      <summary>  </summary> <img src="img/ds/2025-HIVAU70K-anotel.png" width="1500"/>      
      </details>
      <details> 
      <summary>  </summary> <img src="img/ds/2025-HIVAU70K-anotvl.png" width="1500"/>      
      </details>
    </td>
    <td>FL</td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
  </tr>

  <tr>
    <td>2024</td>
    <td>ECVA <a href="https://github.com/fesvhtr/CUVA">:file_folder:</a> <a href="https://github.com/Dulpy/ECVA">:file_folder:</a> </td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/2024-ECVA0.png" width="1500"/>
      <summary>  </summary> <img src="img/ds/2024-ECVA-anomeval.png" width="1500"/>
      <summary>  </summary> <img src="img/ds/2024-ECVA-imprtcrv.png" width="1500"/>
      </details>
    </td> 
    <td>2240/88h</td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td>100 
      <details> 
      <summary>  </summary> <img src="img/ds/2024-ECVA-stats.png" width="1500"/>
      </details>
    </td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/" width="1500"/>
      </details>
    </td>
    <td>TS</td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
  </tr>

  <tr>
    <td>2025</td>
    <td>M-VAE <a href="HTTPS">:file_folder:</a> </td>
    <td>
      <details> 
      <summary>  </summary> based on CUVA Q-A tasks (i.e., timestamp, classification, reason, result, and description tasks)</br>
      1) For abnormal event quadruples:
      from the reason, result, and description tasks in CUVA -> construct initial quadruples through ChatGPT with the instruction: "Please extract the subject, object, and scene of the event based on the responses below" </br> 
      (2) create multiple candidate sets for subjects, objects, and scenes: </br>
      subjects and objects elements, we manually construct a set of around 40 for subjects and objects and filter elements based on this set.  </br>
      For event types elements, we adopt the 11 categories (i.e., Fighting, Animals, Water, Vandalism, Accidents, Robbery, Theft, Pedestrian, Fire, Violations, and Forbidden) from CUVA as the event types. </br>
      For scenes elements, we assign two annotators to classify scenes for each abnormal event
      </details>
      <details> 
      <summary>  </summary> <img src="img/ds/2025-MVAE.png" width="1500"/>
      </details>
    </td> 
    <td>1000/32.50h</td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td></td>
    <td>
      <!-- <details> 
      <summary>  </summary> <img src="img/ds/" width="1500"/>
      </details> -->
    </td>
    <td>FL</td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
  </tr>

  <tr>
    <td>2025</td>
    <td>UCFVL</td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/2025-UCFVL.png" width="1500"/>
      </details>
    </td> 
    <td>1699/88.2h</td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td></td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/2025-UCFVL-anot.png" width="1500"/>
      </details>
    </td>
    <td>TS</td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
  </tr>

  <tr>
    <td>2025</td>
    <td>VANE <a href="HTTPS">:file_folder:</a> </td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/2025-VANE.png" width="1500"/>
      </details>
    </td> 
    <td>325/-</td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td>9</td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/2025-VANE-anot.png" width="1500"/>
      </details>
    </td>
    <td>TS</td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
  </tr>

  <tr>
    <td>2025</td>
    <td>SurveillanceVQA-589K <a href="HTTPS">:file_folder:</a> </td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/2025-SURVEILLANCEVQA589K0.png" width="1500"/>
      <summary>  </summary> <img src="img/ds/2025-SURVEILLANCEVQA589K1.png" width="1500"/>
      </details>
    </td> 
    <td>3030/159h</td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td>18
      <details> 
      <summary>  </summary> UCFC + Fire, Object/People Falling, Pursuit, Water Incidents
      </details>
    </td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/2025-SURVEILLANCEVQA589K-anot.png" width="1500"/>
      </details>
    </td>
    <td>FL</td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
  </tr>

  <tr>
    <td>2025</td>
    <td>
    Vad-Reasoning-SFT </br>
    Vad-Reasoning-RL </br>
    Vad-Reasoning 
    </td>
    <td>
      <details> 
      <summary>  </summary>
      evaluation metrics: QA  2 accuracy, temporal Intersection-over-Union (IoU), GPT-based reasoning score, and classification accuracy
      </details>
      <details> 
      <summary> </summary> <img src="img/ds/2025-VADReason.png" width="1500"/>
      <summary> Annot Proce  </summary> <img src="img/ds/2025-VADReason-anotproc.png" width="1500"/>
      <summary>  </summary> <img src="img/ds/2025-VADReason-stats.png" width="1500"/>
      </details>
    </td> 
    <td>2193/88.3h </br> 6448/272.2h </br> 8641/360.5h </td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td>37</br>VL</br><details> <summary>  </summary> <img src="img/ds/2025-VADReason-anomtype.png" width="1500"/></details> </td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/2025-VADReason-anot.png" width="1500"/>
      </details>
    </td>
    <td>TS</td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
  </tr>

  <tr>
    <td>2025</td>
    <td>VAU-Bench (VAUR1) </td>
    </td>
    <td>
      <details> 
      <summary>  </summary> 
      VAU benchmark is designed for Chain-of-Thought reasoning</br>
      built from MSAD, UCFC and ECVA, enriched with Chain-of-Thought (CoT) annotations.</br>
      For HIVAU-70K_UCFC and ECVA: DeepSeek-V3 -> video-level summaries, QA pairs, and reasoning chains. </br>
      For MSAD, CoT annotations done through: 
        (1) InternVL-8B-MPO generate initial captions/QA/step-step reasoning;</br>
        (2) DeepSeek-V3 verifies/refines</br>
      1.5 million words of fine-grained textual annotations, averaging 337 words per video -> descriptions, reasoning rationales, and MCQ</br>
      split: 2,939 training, 734 validation, and 929 test </br>
      3,700 temporal annotations to support the anomaly grounding task. 
      </details>
      <details> 
      <summary>  </summary> <img src="img/ds/2025-VAUBench.png" width="1500"/>
      </details>
    </td>
    <td>4602/169.1h</td>
    <td><img src="https://img.shields.io/badge/-red" height="20"></td>
    <td></td>
    <td>
      <details> 
      <summary>  </summary> <img src="img/ds/2025-VAUBench-annot.png" width="1500"/>
      </details>
    </td>
    <td>TS</td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
    <td><img src="https://img.shields.io/badge/-_" height="20"></td>
  </tr>

  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td></td>
    <td>TAU-106K <a href="https://github.com/cool-xuan/TABot">:file_folder:</a> </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td></td>
    <td>___<a href="HTTPS">:link:</a> </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</tbody></table>



> [!important]
> **WIP** For a more detailed information/stats about datasets refer to [Awesome-Video-Anomaly-Detection](https://github.com/Junxi-Chen/Awesome-Video-Anomaly-Detection/tree/main) by [@Junxi-Chen](https://github.com/Junxi-Chen)

<!---
UCA [62] only provides clip-level captions, overlooking the understanding of anomalies across longer time spans. 
CUVA [9] and Hawk [45], on the other hand, only offer video-level instruction data, neglecting finer-grained visual perception and anomaly analysis. 
Our proposed HIVAU-70k takes a multi-temporal granularity perspective. It enables progressive and comprehensive learning, from short-term visual perception to long-term anomaly reasoning.

Methods,#Catogories,#Samples,Text(clip-level,event-level,video-level),TempAnno,MLLM-tuning  
UCA,13,23542,✓,✗,✗,✓,✗
LAVAD,N/A,N/A,✓,✗,✓,✗,✗
VAD-VideoLLama,13/7,2400,✗,✗,✓,✗,projection
CUVA,11,6000,✗,✗,✓,✗,✗ 
Hawk/-/16000,✗,✗,✓,✗,projection 
HIVAU-70k, 19,70000,✓,✓,✓,✓,LoRA

-->

# Eval Metrics

<!-- 

% UCFA
    Temporal Sentence Grounding in Videos
    R@K for IoU=θ is commonly adopted as the evaluation metric to measure the performance in TSGV. It is defined as the percentage of at least one of the top-K predicted moments that have IoU with ground-truth moment larger than θ [12]. In the following, we set R@K for IoU= θ with K = 1, 5 and θ = 0.3, 0.5, 0.7 as the evaluation metric.
    Video Captioning
    We use the metrics as in [22, 33]. The evaluation metrics of correctness include Bilingual Evaluation Understudy (BLEU) [B@n, n=1,2,3,4] [29], Metric for Evaluation of Translation with Explicit Ordering (METEOR) [M] [11], Recall Oriented Understudy of Gisting Evaluation (ROUGE-L) [R] [21], and Consensus-based Image Description Evaluation (CIDEr) [C] [40].
    Dense Video Captioning
    We have assessed the performance from two perspectives. For localization performance, we employed the evaluation tools provided by the ActivityNet Captions Challenge 2018 and used the common metrics [43, 50], like different IoU thresholds (0.3, 0.5, 0.7, 0.9), classic caption evaluation metrics: BLEU, METEOR, CIDEr, and the performance in describing video stories: SODA_c score.

    % MSAD
    Micro-AUC and Macro-AUC. The former concatenates the frames from all the videos and computes the overall AUC value, and the latter computes the AUC value for each video and averages it. 
    For pixel-level annotations, Ramachandra et al. [67] introduced two evaluation metrics: the RegionBased Detection Criterion (RBDC) and the Track-Based Detection Criterion (TBDC). These metrics are designed to prioritize the false positive rate on both temporal and spatial dimensions. This consideration stems from the fact that anomalies in videos often extend across multiple frames. Hence, detecting anomalies in any segment and reducing the false detection rate holds significance for VAD systems.
    
    % sherlock/m-vae
    For localization performance, we use the mAP@tIoU metric [71], calculated by mean Average Precision (mAP) at different IoU thresholds from 0.1 to 0.3 with 0.1 intervals. For classification performance, we refer to the traditional anomaly classification task [17, 45, 61] for anomaly classification metric, which mainly determines whether each video frame is abnormal or not in the video. We prefer Recall over Precision and report F2 [71] as another classification metric. Furthermore, our model focuses on accurately distinguishing abnormal events. As shown in Figure 1, it’s better to mark all timestamps as abnormal than to miss any. So we prioritize false negative rates (FNRs): FNRs = num of false-negative frame  num of positive frame , which is the rate of mislabeling an abnormal event frame as normal. In addition, t-test is used to evaluate the significance of the performance.
    
    % PLOVAD 
    For detection, following previous works [1], [4], the framelevel area under the ROC curve(AUC) is adopted as the evaluation metric for UCF-Crime, ShanghaiTech and UBnormal. A higher AUC indicates superior detection performance. For XD-Violence, we utilize AUC of the frame-level precisionrecall curve (AP), following [58].  
    For categorization, the Multi-class AUC for individual classes is computed, and their macro mean, termed mAUC,is derived as the evaluation metric. Multi-class AUC [60] is computed using the one-vs-rest approach, treating each class in turn as positive and others as negative.  mAUC = 1  K  K  X  k=1  AUCk (15)  where K is the number of classes, AUCk is AUC for class k, considering class k as positive and others as negative. Additionally, Top-1 accuracy and Top-5 accuracy are utilized for the evaluation of categorization performance, in accordance with the standard evaluation protocol for video action classification [41].

 -->

# Methodos 
Below are the collected works (mainly) under Weakly supervision (UCFC/XDV), but also new benchamrks.

<u> Recomended</u> to have this extension for redabiliity purposes [wide-github](https://github.com/xthexder/wide-github) + zoom to [60-80]%. 

<details> 
  <summary> legend </summary> 

<!-- 
<a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>
<a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> 
[![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](#)
[![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#)  
-->

- <img src="https://img.shields.io/badge/DataBench-rgb(128, 120, 12)" height="20"> Dataset/Benchmark release paper   

- <img src="https://img.shields.io/badge/Weakly-FFA500" height="20">  Train: Video-Level (VL) labelling, either binary or class // Test: Frame-Level (FL) <!-- Orange -->

- <img src="https://img.shields.io/badge/Glance-F6825E" height="20"> *For each abnormal video in the training set, a randomly chosen single-frame click gi is provided between the start si and end ei of each abnormal instance (si < gi < ei).*

- <img src="https://img.shields.io/badge/Train_Free-808080" height="20"> = Zero-Shot <!-- Gray -->

- <img src="https://img.shields.io/badge/Open_World-00BFFF" height="20"> Evaluation on novel anomalies
<!-- Deep Sky Blue -->

<!-- - <img src="https://img.shields.io/badge/FineTune-rgb(21, 197, 197)" height="20"> -->

- <img src="https://img.shields.io/badge/Explainable-green" height="20"> Provide a mapping of detected visual anomalies to textual descriptions or semantic cues, using either ditionary, LLM's insights or VLM's textual explanations.

- <a href="#"><img src="https://img.shields.io/badge/uws4vad-purple" height="20"></a><img src="https://img.shields.io/badge/done-green" height="20"><img src="https://img.shields.io/badge/wip-yellow" height="20"><img src="https://img.shields.io/badge/roadmap-blue" height="20">
  - Badge is a link to file implementing the method (or better be config since it contains everything).

- Feature in <font color=yellow>yellow</font> means authors have provided download info

- CLIP: if not mentioned refers to ViT B/16
</details> 


<!-- ![PyTorch Badge](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=fff&style=flat) -->

<!--

<details><summary> legend </summary></details>

![](https://img.shields.io/badge/Weakly-FFA500)   
![](https://img.shields.io/badge/Open_Vocabulary-00BFFF) 
![](https://img.shields.io/badge/Train_Free-808080) 
![](https://img.shields.io/badge/Explainable-green) 
[![](https://img.shields.io/badge/uws4vad--green?labelColor=purple)](#)

![](https://img.shields.io/badge/uws4vad-rgba(0,0,0,0)?logo=github&logoColor=green)
-->



## UCFC/XDV

  Results refer to the test subset which methods were evaluated, *o(verall)/a(nomaly)*, when provided. 

  *AUC@ROC* / *AveragePrecision* / *FalseAlarmRate* 

  The *mAP@IoU* is the average value of the mean AP under diff intersection over union (IoU) thresholds ([0.1,0.5] w/ stride=0.1) (multi-class classification)

<table><thead>
  <tr>
    <th>Year</th>
    <th>Method</th>
    <th>Code</th>
    <th>Pipeline &<br>Anom Criterion</th>
    <th>Supervision</th>
    <th>Feature</th>
    <th>UCF <br>(AUC<sub>o</sub>/AUC<sub>a</sub>/FAR/mAP@IoU)</th>
    <th>XDV <br>(AP<sub>o</sub>/AP<sub>a</sub>/FAR/mAP@IoU)</th>
  </tr></thead>
<tbody>
  <tr>
    <td>2018</td>
    <td>MIR <a href="https://ieeexplore.ieee.org/document/8578776/"> <img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a> </br> 
      <img src="https://img.shields.io/badge/UCFC-rgb(128, 120, 12)" height="20">   
    </td>
    <td> 
      <a href=""><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>
      <img src="https://img.shields.io/badge/uws4vad--green?labelColor=purple" height="20">
    </td>
    <td> <details> <summary></summary> <img src="img/meth/1-2018-MIR.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D<br>I3D</td>
    <td>75.41<br>77.92</td>
    <td><br></td>
  </tr>
  <tr>
    <td>2019</td>
    <td>GCN <a href="https://ieeexplore.ieee.org/abstract/document/8953791"> <img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a></td>
    <td> <a href="https://github.com/jx-zhong-for-academic-purpose/GCN-Anomaly-Detection"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> </td>
    <td> <details> <summary></summary> <img src="img/meth/2-2019-GCNVAD.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D<br>TSN</td>
    <td>81.92<br>82.12</td>
    <td><br></td>
  </tr>
  <tr>
    <td>2019</td>
    <td>MA <a href="#"> <img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>PWC-OF</td>
    <td>72.10</td>
    <td></td>
  </tr>
  <tr>
    <td>2019</td>
    <td>TCN <a href="#"> <img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td> <details> <summary></summary> <img src="img/meth/3-2019-TCNIBL.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D </td>
    <td>78.66</td>
    <td></td>
  </tr>
  <tr>
    <td>2020</td>
    <td>SRF <a href="#"> <img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td> <details> <summary></summary> <img src="img/meth/5-2020-SRF.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D </td>
    <td>79.54/-/0.13</td>
    <td></td>
  </tr>
  <tr>
    <td>2020</td>
    <td>ARN <a href="http://arxiv.org/abs/2104.07268"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td> 
    <td><a href="https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020"> <img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a></td>
    <td> </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D </td>
    <td>75.71</td>
    <td></td>
  </tr>
  <tr>
    <td>2020</td>
    <td>HLN 
      <a href="https://roc-ng.github.io/XD-Violence/images/paper.pdf"><img src="https://img.shields.io/badge/ECCV-rgba(0,0,0,0)" height="20"></a> </br>
      <img src="https://img.shields.io/badge/XDV-rgb(128, 120, 12)" height="20">
    </td>
    <td><a href="https://github.com/Roc-Ng/XDVioDet"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a></td>
    <td> <details> <summary></summary> <img src="img/meth/6-2020-XDVHLNET.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D<br><font color="yellow">I3D+VGG</font> </td> 
    <td>82.44<br>-</td>
    <td>-<br>78.64</td>
  </tr>
  <tr>
    <td>2020</td>
    <td>WSAL <a href="https://ieeexplore.ieee.org/abstract/document/9408419"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a></td>
    <td> <a href="https://github.com/ktr-hubrt/WSAL"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> </td>
    <td> <details> <summary></summary> <img src="img/meth/7-2020-WSAL.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D <br>R(2+1)D<br>TSN</td>
    <td>74.18 <br> 75.29 <br> 85.38/67.38/-</td>
    <td> <br>  <br> </td>
  </tr>
  <tr>
    <td>2020</td>
    <td>CLAWS <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td> <details> <summary></summary> <img src="img/meth/8-2020-CLAWS.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D</td>
    <td>83.03/-/0.12</td>
    <td></td>
  </tr>
  <tr>
    <td>2021</td>
    <td>MIST <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td> <details> <summary></summary> <img src="img/meth/9-2021-MIST.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D <br>I3D</td>
    <td>81.40/-/2.19 <br> 82.30/-/0.13</td>
    <td> <br> </td>
  </tr>
  <tr>
    <td>2021</td>
    <td>AVF <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td> <details> <summary></summary> <img src="img/meth/10-2021-AVF.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D+VGG</td>
    <td></td>
    <td>81.69</td>
  </tr>
  <tr>
    <td>2021</td>
    <td>RTFM <a href="http://arxiv.org/abs/2101.10030"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td> 
      <a href="https://github.com/tianyu0207/RTFM"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> 
      <img src="https://img.shields.io/badge/uws4vad--green?labelColor=purple" height="20"> 
    </td>
    <td> <details> <summary></summary> <img src="img/meth/11-2021-RTFM.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D <br>I3D</td>
    <td>83.28 <br> 84.30</td>
    <td>75.89 <br> 77.81</td>
  </tr>
  <tr>
    <td>2021</td>
    <td>XEL <a href="https://ieeexplore.ieee.org/document/9560033/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a> </td>
    <td>
      <a href="https://github.com/sdjsngs/XEL-WSAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>
      <img src="https://img.shields.io/badge/uws4vad--yellow?labelColor=purple" height="20">  
    </td>
    <td> <details> <summary></summary> <img src="img/meth/12-2021-XEL.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D</td>
    <td>82.60</td>
    <td></td>
  </tr>
  <tr>
    <td>2021</td>
    <td>CA  <a href="https://ieeexplore.ieee.org/document/9540293"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a></td>
    <td><a href="https://github.com/changsn/Contrastive-Attention-for-Video-Anomaly-Detection"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a></td>
    <td> <details> <summary></summary> <img src="img/meth/13-2021-CA.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D <br>TSN <br>I3D </td>
    <td>83.40 <br> 83.52 <br> 84.62</td>
    <td> <br>  <br> 76.90</td>
  </tr>
  <tr>
    <td>2021</td>
    <td>MS-BS  <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td> <details> <summary></summary> <img src="img/meth/14-2021-MSBS.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D</td>
    <td>83.53</td>
    <td></td>
  </tr>
  <tr>
    <td>2021</td> 
    <td>DAM <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td>
      <a href="https://github.com/snehashismajhi/DAM-Anomaly-Detection"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=tensorflow" height="20"></a>
    </td>
    <td> <details> <summary></summary> <img src="img/meth/15-2021-DAM.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D</td>
    <td>82.67/-/0.3</td>
    <td></td>
  </tr>
  <tr>
    <td>2022</td>
    <td>CMALA <a href="https://ieeexplore.ieee.org/document/9712793"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a> 
    <td>
      <a href="https://github.com/yujiangpu20/cma_xdVioDet"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> 
      <img src="https://img.shields.io/badge/uws4vad--green?labelColor=purple" height="20"></td>
    </td>
    <td> <details> <summary></summary> <img src="img/meth/16-2022-CMALA.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D+VGG</td>
    <td></td>
    <td>83.54</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>CLAWS+ <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D <br>3DRN </td>
    <td>83.37/-/0.11 <br> 84.16/-/0.09</td>
    <td> <br> </td>
  </tr>
  <tr>
    <td>2022</td>
    <td>WSTR <a href="https://ieeexplore.ieee.org/abstract/document/9774889"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a></td>
    <td><a href="https://github.com/justsmart/WSTD-VAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> </td>
    <td> <details> <summary></summary> <img src="img/meth/17-2022-WSTR.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D </td>
    <td>83.17</td>
    <td></td>
  </tr>
  <tr>
    <td>2022</td>
    <td>STA <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td> <details> <summary></summary> <img src="img/meth/18-2022-STA.png width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D<br>I3D</td>
    <td>81.60 <br> 83.00</td>
    <td> <br> </td>
  </tr>
  <tr>
    <td>2022</td>
    <td>MSL <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td><a href="https://github.com/LiShuo1001/MSL"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> </td>
    <td> <details> <summary></summary> <img src="img/meth/19-2022-MSL.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D <br>I3D <br>VSWIN</td>
    <td>82.85 <br> 85.30 <br> 85.62</td>
    <td>75.53 <br> 78.28 <br> 78.59</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>SGMIR <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td> <details> <summary></summary> <img src="img/meth/20-2022-SGMIR.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D</td>
    <td>81.70</td>
    <td></td>
  </tr>
  <tr>
    <td>2022</td>
    <td>TCA <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td> <details> <summary></summary> <img src="img/meth/21-2022-TCAVAD.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D <br>I3D</td>
    <td>82.08/-/0.11 <br> 83.75/-/0.05</td>
    <td> <br> </td>
  </tr>
  <tr>
    <td>2022</td>
    <td>MACILSD <a href="http://arxiv.org/abs/2207.05500"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td>
      <a href="https://github.com/JustinYuu/MACIL_SD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> 
      <img src="https://img.shields.io/badge/uws4vad--yellow?labelColor=purple" height="20">
    </td>
    <td> <details> <summary></summary> <img src="img/meth/22-2022-MACILSD.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D+VGG </td>
    <td></td>
    <td>83.40</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>LAN <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td> <details> <summary></summary> <img src="img/meth/23-2022-LAN.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D </td>
    <td>85.12</td>
    <td>80.72</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>OpenVAD  <a href="https://arxiv.org/abs/2208.11113"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td><a href="https://github.com/YUZ128pitt/Towards-OpenVAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a></td>
    <td> 
      <img src="https://img.shields.io/badge/Open-00BFFF" height="20">
      <details> <summary></summary> <img src="img/meth/24-2022-OPENVAD.png" width="1500"/> </details>  </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2022</td>
    <td>ANM <a href="http://arxiv.org/abs/2209.06435"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td>
      <a href="https://github.com/sakurada-cnq/salient_feature_anomaly"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> 
      <img src="https://img.shields.io/badge/uws4vad--green?labelColor=purple" height="20">
    </td>
    <td> <details> <summary></summary> <img src="img/meth/25-2022-ANMIL.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D <br>I3D+VGG</td>
    <td>82.99 <br> </td>
    <td> <br> 84.91</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>MSAF  <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td><a href="https://github.com/Video-AD/MSFA"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a></td>
    <td> <details> <summary></summary> <img src="img/meth/26-2022-MSAF.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D+OF <br>I3D+OF+VGG</td>
    <td>81.34 <br> </td>
    <td> <br> 80.51</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>TAI <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td> <details> <summary></summary> <img src="img/meth/27-2022-TAI.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D</td>
    <td>85.73</td>
    <td></td>
  </tr>
  <tr>
    <td>2022</td>
    <td>BSME <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td> <details> <summary></summary> <img src="img/meth/28-2022-BSME.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D</td>
    <td>83.63</td>
    <td></td>
  </tr>
  <tr>
    <td>2022</td>
    <td>MGFN <a href="https://arxiv.org/abs/2211.15098"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td>
      <a href="https://github.com/carolchenyx/MGFN."><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> 
      <img src="https://img.shields.io/badge/uws4vad--yellow?labelColor=purple" height="20">
    </td>
    <td> <details> <summary></summary> <img src="img/meth/29-2022-MGFN.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D <br>VSwin</td>
    <td>86.98 <br> 86.67</td>
    <td>79.19 <br> 80.11</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>CUN <a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Exploiting_Completeness_and_Uncertainty_of_Pseudo_Labels_for_Weakly_Supervised_CVPR_2023_paper.pdf"><img src="https://img.shields.io/badge/CVPR-rgba(0,0,0,0)" height="20"></a></td>
    <td><a href="https://github.com/ArielZc/CU-Net"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> </td>
    <td> <details> <summary></summary> <img src="img/meth/30-2022-CUN.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D <br>I3D+VGG </td>
    <td>86.22 <br> </td>
    <td>78.74 <br> 81.43</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>CLIP-TSA  <a href="https://arxiv.org/abs/2212.05136"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td>
      <a href="https://github.com/joos2010kj/CLIP-TSA"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>
      <img src="https://img.shields.io/badge/uws4vad--yellow?labelColor=purple" height="20">
    </td>
    <td> <details> <summary></summary> <img src="img/meth/31-2022-CLIPTSA.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td style="color:yellow">CLIP</td>
    <td>87.58</td>
    <td>82.19</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>NGMIL <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td> <details> <summary></summary> <img src="img/meth/32-2023-NGMIL.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D <br>I3D </td>
    <td>83.43 <br> 85.63</td>
    <td>75.91 <br> 78.51</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>URDMU  <a href="http://arxiv.org/abs/2302.05160"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td>
      <a href="https://github.com/henrryzh1/UR-DMU"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>
      <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20">
    </td>
    <td> <details> <summary></summary> <img src="img/meth/33-2023-URDMU.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D <br>I3D+VGG</td>
    <td>86.97/-/1.05 <br> </td>
    <td>81.66/-/0.65 <br> 81.77</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>UMIL <a href="http://arxiv.org/abs/2303.12369"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td><a href="https://github.com/ktr-hubrt/UMIL"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> </td>
    <td> <details> <summary></summary> <img src="img/meth/34-2023-UMIL.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>X-CLIP</td>
    <td>86.75/68.68/-</td>
    <td></td>
  </tr>
  <tr>
    <td>2023</td>
    <td>LSTC  <a href="http://arxiv.org/abs/2303.18044"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td><a href="https://github.com/shengyangsun/LSTC_VAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a></td>
    <td> <details> <summary></summary> <img src="img/meth/35-2023-LSTC.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D<br>I3D </td>
    <td>83.47 <br> 85.88</td>
    <td> <br> </td>
  </tr>
  <tr>
    <td>2023</td>
    <td>BERTMIL <a href="https://arxiv.org/abs/2210.06688"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td><a href="https://github.com/wjtan99/BERT_Anomaly_Video_Classification"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> </td>
    <td> <detaIls> <summary></summary> <img src="img/meth/36-2023-BERTMILRTFM.png" width="600"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D+FLOW <br>I3D </td>
    <td>86.71 <br> </td>
    <td> <br> 82.10</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>TEVAD <a href="https://ieeexplore.ieee.org/document/10208872/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a></td>
    <td><a href="https://github.com/coranholmes/TEVAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> </td>
    <td> <details> <summary></summary> <img src="img/meth/37-2023-TEVAD.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2023</td>
    <td>HYPERVD  <a href="http://arxiv.org/abs/2305.18797"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td><a href="https://github.com/xiaogangpeng/HyperVD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a></td>
    <td> <details> <summary></summary> <img src="img/meth/39-2023-HYPERVD.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D+VGG </td>
    <td></td>
    <td>85.67</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>PEL4VAD  <a href="http://arxiv.org/abs/2306.14451"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td>
      <a href="https://github.com/yujiangpu20/PEL4VAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>
      <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20"> 
    </td>
    <td> <details> <summary></summary> <img src="img/meth/40-2023-PEL4VAD.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D </td>
    <td>86.76/72.24/0.43</td>
    <td>85.59/70.26/0.57</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>VAR  <a href="http://arxiv.org/abs/2307.12545"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td> <a href="https://github.com/Roc-Ng/VAR"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> </td>
    <td> <details> <summary></summary> <img src="img/meth/41-2023-VAR.png" width="1500"/> </details>  </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2023</td>
    <td>CNN-VIT <a href="https://www.mdpi.com/1424-8220/23/18/7734"><img src="https://img.shields.io/badge/MDPI-rgba(0,0,0,0)" height="20"></a></td>
    <td></td>
    <td> <details> <summary></summary> <img src="img/meth/42-2023-CNNVIT.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D <br>I3D <br>CLIP <br>C3D+CLIP<br>I3D+CLIP </td>
    <td>85.78 <br> 86.50 <br> 87.63 <br> 88.02 <br> 88.97</td>
    <td> <br>  <br>  <br>  <br> </td>
  </tr>
  <tr>
    <td>2024</td>
    <td>TeD-SPAD  <a href="http://arxiv.org/abs/2308.11072"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td>
      <a href="https://github.com/UCF-CRCV/TeD-SPAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>
      <a href="https://joefioresi718.github.io/TeD-SPAD_webpage/">:link:</a>
    </td>
    <td> <details> <summary></summary> <img src="img/meth/43-2023-TEDSPAD.png" width="1500"/> </details>  </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr> 
  <tr>
    <td>2023</td>
    <td>SAA <a href="http://arxiv.org/abs/2309.16309"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td>
      <a href="https://github.com/YukiFan/vad-weakly"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> 
      <a href="https://github.com/Daniel00008/WS-VAD-mindspore"><img src="https://img.shields.io/badge/MS-rgba(0,0,0,0)" height="20"></a> 
      <a href="https://github.com/2023-MindSpore-4/Code4"><img src="https://img.shields.io/badge/MS-rgba(0,0,0,0)" height="20"></a> </br>
      <img src="https://img.shields.io/badge/uws4vad--yellow?labelColor=purple" height="20"> 
    </td>
    <td> <details> <summary></summary> <img src="img/meth/44-2023-SAA.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D <br>I3D+VGG </td>
    <td>86.19/68.77/- <br> </td>
    <td>83.59/84.19/- <br> 84.23</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>AnomalyCLIP <a href="http://arxiv.org/abs/2310.02835"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td> <a href="https://github.com/lucazanella/AnomalyCLIP"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> </td>
    <td> <details> <summary></summary> <img src="img/meth/45-2023-ANOMALYCLIP.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>ViT-B/16 </td>
    <td>86.36</td>
    <td>78.51</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>MTDA <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td> <details> <summary></summary> <img src="img/meth/46-2023-MTDA.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D+VGG </td>
    <td></td>
    <td>84.44</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>BNWVAD  <a href="http://arxiv.org/abs/2311.15367"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td>
      <a href="https://github.com/cool-xuan/BN-WVAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>
      <img src="https://img.shields.io/badge/uws4vad--yellow?labelColor=purple" height="20">
    </td>
    <td> <details> <summary></summary> <img src="img/meth/47-2023-BNDFM.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D <br>I3D+VGG </td>
    <td>87.24/71.71/- <br> </td>
    <td>84.93/85.45/- <br> 85.26</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>DEN  <a href="http://arxiv.org/abs/2312.01764"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td>
      <a href="https://github.com/ArielZc/DE-Net"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>
      <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20"> 
    </td>
    <td> <details> <summary></summary> <img src="img/meth/48-2023-DEN.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D <br>I3D+VGG </td>
    <td>86.33 <br> </td>
    <td>81.66 <br> 83.13</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>VADCLIP  <a href="http://arxiv.org/abs/2308.11681"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a>  </td>
    <td>
      <a href="https://github.com/nwpu-zxr/VadCLIP"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>
      <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20">
    </td>
    <td>
      <details> <summary></summary> <img src="img/meth/49-2023-VADCLIP.png" width="1500"/> </details>  
    </td>
    <td>
      <img src="https://img.shields.io/badge/Weak-FFA500" height="20">
    </td>
    <td style="color:yellow">CLIP</td>
    <td>88.02/70.23/-/6.68</td>
    <td>84.51/-/-/24.70</td>
  </tr>
  <tr>
    <td>2024</td>
    <td>REWARD
      <a href="http://arxiv.org/abs/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20">
      <td> <a href="https://github.com/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a></td>
    </td>
    <td> <details> <summary></summary> <img src="img/meth/50-2024-REWARD.png" width="1500"/> </details></td>
    <td>
      <img src="https://img.shields.io/badge/Weak-FFA500" height="20">
    </td>
    <td></td>
    <td></td>
    <td></td>
  </tr> 
  <tr>
    <td>2024</td>
    <td>VAD-LLaMA  <a href="http://arxiv.org/abs/2401.05702"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td> <a href="https://github.com/ktr-hubrt/VAD-LLaMA"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> </td>
    <td> 
      <img src="https://img.shields.io/badge/Explainable-green" height="20">
      <details> <summary></summary> <img src="img/meth/51-2024-VADLLAMA.png" width="1500"/> </details>  
    </td>
    <td>
      <img src="https://img.shields.io/badge/Weak-FFA500" height="20">
    </td>
    <td>Video-LLaMA(I+T)</td>
    <td>88.13/71.12/-</td>
    <td></td>
  </tr>
  <tr>
    <td>2024</td>
    <td>LAP <a href="http://arxiv.org/abs/2403.01169"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td> <details> <summary></summary> <img src="img/meth/52-2024-LAP.png" width="1500"/> </details>  </td>
    <td>
      <img src="https://img.shields.io/badge/Weak-FFA500" height="20">
    </td>
    <td>CLIP+SwinBERT</td>
    <td>87.7/71.1</td>
    <td>82.6</td>
  </tr>
  <tr>
    <td>2024</td>
    <td> <img src="https://img.shields.io/badge/GlanceVAD-rgb(128, 120, 12)" height="20"><a href="http://arxiv.org/abs/2403.06154"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td> 
      <a href="https://github.com/pipixin321/GlanceVAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> 
      <img src="https://img.shields.io/badge/uws4vad--green?labelColor=purple" height="20">
    </td>
    <td> 
      <details> <summary></summary> <img src="img/meth/53-2024-GLANCEVAD.png" width="1500"/> </details>  
    </td>
    <td>
      <img src="https://img.shields.io/badge/Glance-F6825E" height="20">
    </td>
    <td>I3D</td>
    <td>MIL 87.30/71.07 </br> RTFM 87.80/75.16 </br> URDMU 91.96/84.94</td>
    <td>MIL 83.61/86.15 </br> RTFM 86.88/86.65 </br> URDMU 89.40/89.85</td>
  </tr>
    <tr>
    <td>2024 </td>
    <td>OVVAD <a href="http://arxiv.org/abs/2311.07042"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td> 
      <img src="https://img.shields.io/badge/Open-00BFFF" height="20"> 
      <details> <summary></summary> <img src="img/meth/54-2024-OVVAD.png" width="1500"/> </details>  
    </td>
    <td> <img src="https://img.shields.io/badge/Weak-FFA500" height="20"> </td>
    <td>CLIP</td>
    <td>86.40/88.20/-</td>
    <td>66.53/76.03/-</td>
  </tr>
  <tr>
    <td>2024</td>
    <td>LAVAD <a href="https://arxiv.org/abs/2404.01014"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td> <a href="https://github.com/lucazanella/lavad"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>  </td>
    <td>
      <img src="https://img.shields.io/badge/Open-00BFFF" height="20"> </br>
      <img src="https://img.shields.io/badge/Explainable-green" height="20"> 
      <details> <summary></summary> <img src="img/meth/55-2024-LAVAD.png" width="1100"/> </details> 
    </td>
    <td><img src="https://img.shields.io/badge/Train_Free-808080" height="20"></td>
    <td> BLIP-2 ensmb <br> Llama-2-13b-chat <br> ImageBind MM enc. </td>
    <td>80.28</td>
    <td>62.01</td>
  </tr>
  <tr>
    <td>2024</td>
    <td>TPWNG <a href="http://arxiv.org/abs/2404.08531"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td>
      <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20">
    </td>
    <td> <details> <summary></summary> <img src="img/meth/56-2024-TPWNG.png" width="1500"/> </details>  </td>
    <td>
      <img src="https://img.shields.io/badge/Weak-FFA500" height="20">
    </td>
    <td>CLIP(I+T)</td>
    <td>87.79</td>
    <td>83.68</td>
  </tr>
  <tr>
    <td>2024</td>
    <td>MSBT <a href="https://arxiv.org/abs/2405.05130"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td> <a href="https://github.com/shengyangsun/MSBT"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> </td>
    <td> <details> <summary></summary> <img src="img/meth/58-2024-MSBT.png" width="1500"/> </details>  </td>
    <td>
      <img src="https://img.shields.io/badge/Weak-FFA500" height="20">
    </td>
    <td>I3D+VGG <br>I3D+VGG+TV-L1 </td>
    <td> <br> </td>
    <td>82.54 <br> 84.32</td>
  </tr>
  <tr>
    <td>2024</td>
    <td> 
      <img src="https://img.shields.io/badge/HAWK-rgb(128, 120, 12)" height="20"> <a href="http://arxiv.org/abs/2405.16886">
      <img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a>
    </td>
    <td> 
      <a href="https://github.com/jqtangust/hawk"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> 
      <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20">
    </td>
    <td> 
      <img src="https://img.shields.io/badge/Explainable-green" height="20"></br>
      <details> <summary></summary> <img src="img/meth/59-2024-HAWK.png" width="1500"/> </details>  
    </td>
    <td><img src="https://img.shields.io/badge/Instr-Tuned-green" height="20"></td>
    <td>BLIP2</br>EVACLIP+QFORMER</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2024</td>
    <td>PEMIL  <a href="https://ieeexplore.ieee.org/document/10657732/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a> </td>
    <td>
      <a href="https://github.com/Junxi-Chen/PE-MIL"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>
      <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20">
    </td>
    <td>
      <details> <summary></summary> <img src="img/meth/60-2024-PEMIL.png" width="1500"/> </details> 
    </td>
    <td>
      <img src="https://img.shields.io/badge/Weak-FFA500" height="20">
    </td>
    <td>I3D<br>I3D+VGG</td>
    <td>86.83</td>
    <td>88.05 </br> 88.21</td>
  </tr>
  <tr>
    <td>2024</td>
    <td><img src="https://img.shields.io/badge/UCF-A -rgb(128, 120, 12)" height="20">  <a href="https://ieeexplore.ieee.org/document/10656129/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a></td>
    <td>
       <a href="https://github.com/Xuange923/Surveillance-Video-Understanding"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> 
    </td>
    <td> 
      <img src="https://img.shields.io/badge/VideoCaption-rgba(0,0,0,0)" height="20"> 
      <details> <summary></summary> <img src="img/meth/61-2024-UCFA.png" width="1500"/> </details>          
    </td>
    <td> </td> 
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr> 
  <tr>
    <td>2024</td>
    <td>Holmes-VAD 
      <a href="http://arxiv.org/abs/2406.12235"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> 
      <a href="https://holmesvad.github.io/"> :link: </a> 
    </td>
    <td> <a href="https://github.com/pipixin321/HolmesVAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a></td>
    <td> 
      <img src="https://img.shields.io/badge/Explainable-green" height="20"></br>
      <details> <summary></summary> <img src="img/meth/62-2024-HOLMESVAD.png" width="1500"/> </details>  
    </td>
    <td>  
      <img src="https://img.shields.io/badge/Glance-F6825E" height="20"> +</br>
      <img src="https://img.shields.io/badge/Instr-Tuned-green" height="20">
    </td>
    <td>LanguageBind+</br>URDMU+</br>VideoLLaVA</br></td>
    <td>89.51</td>
    <td>90.67</td>
  </tr>
  <tr>  
    <td>2024</td>
    <td>Holmes-VAU  
      <a href="http://arxiv.org/abs/2412.06171"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a>  </br>
      <img src="https://img.shields.io/badge/HIVAU70k-rgb(128, 120, 12)" height="20">
    </td>
    <td>
      <a href="https://github.com/pipixin321/HolmesVAU"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>
      <img src="https://img.shields.io/badge/uws4vad--yellow?labelColor=purple" height="20">
    </td>
    <td>
      <img src="https://img.shields.io/badge/Explainable-green" height="20"></br>
      <details> <summary></summary> <img src="img/meth/67-2024-HOLMESVAU.png" width="1500"/> </details>
    </td>
    <td>  
      <img src="https://img.shields.io/badge/Super(FL)-F4511E" height="20"> +</br>
      <img src="https://img.shields.io/badge/Instr-Tuned-green" height="20">
    </td>
    <td>InternVL2-2B</br>(I+T)</td>
    <td>88.96</td>
    <td>87.68</td>
  </tr>
  <tr>
    <td>2024</td>
    <td>FE-VAD <a href="https://ieeexplore.ieee.org/document/10688326/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a></td>
    <td>
      <a href="https://github.com/pipixin321/HolmesVAU"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>
      <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20">
    </td>
    <td> <details> <summary></summary> <img src="img/meth/63-2024-FEVAD.png" width="1500"/> </details>   </td>
    <td>
      <img src="https://img.shields.io/badge/Weak-FFA500" height="20">
    </td>
    <td>I3D</td>
    <td>87.13</td>
    <td>82.87</td>
  </tr>
  <tr>
    <td>2024</td>
    <td>STPrompt
      <a href="http://arxiv.org/abs/2408.05905/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> 
    </td>
    <td><img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20"> </td>
    <td> <details> <summary></summary> <img src="img/meth/64-2024-STPROMPT.png" width="1500"/> </details></td>
    <td>
      <img src="https://img.shields.io/badge/Weak-FFA500" height="20">
    </td>
    <td>CLIP(I+T)</td>
    <td>88.08</td>
    <td></td>
  </tr>
  <tr>
    <td>2024</td>
    <td>ITC
      <a href="https://ieeexplore.ieee.org/document/10719608"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a> 
    </td>
    <td><img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20"></td>
    <td> <details> <summary></summary> <img src="img/meth/65-2024-ITC.png" width="1500"/> </details></td>
    <td>
      <img src="https://img.shields.io/badge/Weak-FFA500" height="20">
    </td>
    <td>CLIP(I+T)</td>
    <td>89.04/-/-/7.90</td>
    <td>85.45/-/-/26.83</td>
  </tr>
  <tr>
    <td>2024</td>
    <td>TDSD
      <a href="https://dl.acm.org/doi/10.1145/3664647.3680934"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=aCM&logoColor=white" height="20"></a> 
    </td>
    <td> <a href="https://github.com/shengyangsun/TDSD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> </td>
    <td> <details> <summary></summary> <img src="img/meth/66-2024-TDSD.png" width="1500"/> </details></td>
    <td>
      <img src="https://img.shields.io/badge/Weak-FFA500" height="20">
    </td>
    <td>CLIP(I+T)</br>(ViT-L/14)</td>
    <td></td>
    <td>84.69</td>
  </tr>
  <tr>
    <td>2024</td>
    <td> <img src="https://img.shields.io/badge/MSAD-rgb(128, 120, 12)" height="20">
      <a href="http://arxiv.org/abs/2402.04857"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a>
    </td>
    <td>
      <a href="https://github.com/Tom-roujiang/MSAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> 
      <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20">
    </td>
    <td> 
      <!-- <details> <summary></summary> <img src="img/ds/2024-MSAD.png" width="1500"/> </details> -->
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2024</td>
    <td>AnomShield 
      <a href="http://arxiv.org/abs/2412.07183"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </br>
      <img src="https://img.shields.io/badge/ECVA-rgb(128, 120, 12)" height="20">
    </td>
    <td>
      <a href="https://github.com/Dulpy/ECVA"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> 
      <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20">
    </td>
    <td>
      <img src="https://img.shields.io/badge/Explainable-green" height="20">
      <details> <summary></summary> <img src="img/meth/68-2024-ECVA.png" width="1500"/> </details>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
    <tr>
    <td>2024</td>
    <td>QuoVADis <a href="http://arxiv.org/abs/2412.18298"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2025</td>
    <td>AVCL
      <a href="https://ieeexplore.ieee.org/document/10855604/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a> 
    </td>
    <td>
      <!-- <a href="https://github.com/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>  -->
      <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20">
    </td>
    <td> 
      <img src="https://img.shields.io/badge/Open-00BFFF" height="20"> </br>
      <details> <summary></summary> <img src="img/meth/69-2025-AVCL.png" width="1500"/> </details>
    </td>
    <td>
      <img src="https://img.shields.io/badge/Weak-FFA500" height="20">
    </td>
    <td>I3D+VGG</br>I3D+VGG(+MACILSD)</td>
    <td></td>
    <td>81.11</br>83.98</td>
  </tr>
  <tr>
    <td>2025</td>
    <td>PLOVAD <a href="https://github.com/ctX-u/PLOVAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://ieeexplore.ieee.org/document/10836858"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a> </td>
        <td></td>
    <td> 
      <img src="https://img.shields.io/badge/Open-00BFFF" height="20"> </br>
      <!-- <img src="https://img.shields.io/badge/Classfctn-229" height="20"> -->
      <details> <summary></summary> <img src="img/meth/70-2025-PLOVAD.png" width="1500"/> </details>  
    </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>CLIP(I+T)</td>
    <td>87.06</td>
    <td></td>
  </tr>
    <tr>
    <td>2024</td>
    <td> <img src="https://img.shields.io/badge/UCFVL-rgb(128, 120, 12)" height="20">
      <a href="http://arxiv.org/abs/2502.09325"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a>
      <!-- <a href="https://msad-dataset.github.io/"> :link: </a> -->
    </td>
    <td>
      <!-- <a href="https://github.com/Tom-roujiang/MSAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>  -->
      <!-- <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20"> -->
    </td>
    <td> 
      <!-- <details> <summary></summary> <img src="img/ds/.png" width="1500"/> </details> -->
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2025</td>
    <td>MTFL </a> <a href="https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13517/3055069/MTFL--multi-timescale-feature-learning-for-weakly-supervised-anomaly/10.1117/12.3055069.full"><img src="https://img.shields.io/badge/ICMV-rgba(0,0,0,0)" height="20"></a> </td>
    <td></td>
    <td> <details> <summary></summary> <img src="img/meth/71-2025-MTFL.png" width="1500"/> </details> </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>VST</br>VSTAug</td>
    <td>87.16</br>89.78</td>
    <td>84.57</br>79.40</td>
  </tr>
  <tr>
    <td>2025</td>
    <td> <img src="https://img.shields.io/badge/Sherlock-rgb(128, 120, 12)" height="20"> <a href="http://arxiv.org/abs/2502.18863"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td> 
    </td>
    <td>
      <img src="https://img.shields.io/badge/Explainable-green" height="20"> 
      <details> <summary></summary> <img src="img/meth/72-2025-SHERLOCK.png" width="1500"/> </details>  </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2025</td>
    <td> MSTAgentVAD
      <a href="https://www.sciencedirect.com/science/article/pii/S0957417425007766"><img src="https://img.shields.io/badge/SciDir-rgba(0,0,0,0)" height="20">
      <td> </td>
    </td>
    <td> <details> <summary></summary> <img src="img/meth/73-2025-MSTAGENTVAD.png" width="1500"/> </details></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>VidSwinT</td>
    <td>89.27/-/0.0017</td>
    <td></td>
  </tr>
  <tr>
    <td>2025</td>
    <td> <img src="https://img.shields.io/badge/UCF-DVS -rgb(128, 120, 12)" height="20"> <a href="http://arxiv.org/abs/2503.12905"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td> 
      <a href="https://github.com/YBQian-Roy/UCF-Crime-DVS"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>
      <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20">
    </td>
    <td> <details> <summary></summary> <img src="img/meth/74-2025-UCFCDVS.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>Spikingformer</br>(PT@HarDVS)</td>
    <td>65.01/-/3.27</td>
    <td></td>
  </tr>
  <tr>
    <td>2025</td>
    <td><img src="https://img.shields.io/badge/VANE_BENCH -rgb(128, 120, 12)" height="20">
      <a href="http://arxiv.org/abs/2406.10326"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> 
    </td>
    <td><a href="https://github.com/rohit901/VANE-Bench"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> </td>
    <td> 
      <!-- <details> <summary></summary> <img src="img/ds/2025-VANE.png" width="1500"/> </details> -->
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2025</td>
    <td>VADMamba 
      <a href="http://arxiv.org/abs/2503.21169"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a>
    </td>
    <td>
      <a href="https://github.com/jLooo/VADMamba"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> 
    </td>
    <td> 
      <details> <summary></summary> <img src="img/meth/75-2025-VADMAMBA.png" width="1500"/> </details>
    </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>2025</td>
    <td>VERA<a href="http://arxiv.org/abs/2412.01095"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td>  <a href="https://github.com/vera-framework/VERA"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>  </td>
    <td> 
      <img src="https://img.shields.io/badge/Explainable-green" height="20">
      <details> <summary></summary> <img src="img/meth/76-2025-VERA0.png" width="1500"/> </details>   
      <details> <summary></summary> <img src="img/meth/76-2025-VERA1.png" width="1500"/> </details> 
      <details> <summary></summary> <img src="img/meth/76-2025-VERA2.png" width="1500"/> </details> 
    </td>
    <td>
      <img src="https://img.shields.io/badge/Weak-FFA500" height="20">
    </td>
    <td>InternVL2-8B</td>
    <td>86.55</td>
    <td>70.54</td>
  </tr>
    <tr>
    <td>2025</td>
    <td>MELOW
      <a href="https://ieeexplore.ieee.org/document/10948323/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a> 
    </td>
    <td>
      <!-- <a href="https://github.com/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>  -->
      <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20">
    </td>
    <td> 
      <img src="https://img.shields.io/badge/Open-00BFFF" height="20"> </br>
      <details> <summary></summary> <img src="img/meth/77-2025-MELOWVAD.png.png" width="1500"/> </details>
    </td>
    <td>
      <img src="https://img.shields.io/badge/Weak-FFA500" height="20">
    </td>
    <td>CLIP(I+T)</td>
    <td>87.80</td>
    <td>85.13</td>
  </tr>
  <tr>
    <td>2025</td>
    <td>AVadCLIP <a href="http://arxiv.org/abs/2504.04495"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td> 
      <!-- <a href="https://github.com/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> -->
      <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20">
    </td>
    <td> 
      <details> <summary></summary> <img src="img/meth/78-2025-AVADCLIP.png" width="1500"/> </details> 
    </td>
    <td>
      <img src="https://img.shields.io/badge/Weak-FFA500" height="20">
    </td>
    <td>CLIP(I+T)</br>CLIP(I+T)+Wav2CLIP</td>
    <td></td>
    <td>85.53/-/-/27.44</br>86.04/-/-/28.61</td>
  </tr>
  <tr>
    <td>2025</td>
    <td>EventVAD <a href="http://arxiv.org/abs/2504.13092"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td></td>
    <td>
      <img src="https://img.shields.io/badge/Open-00BFFF" height="20"> </br>
      <img src="https://img.shields.io/badge/Explainable-green" height="20">
      <details> <summary></summary> <img src="img/meth/79-2025-EVENTVAD.png" width="1500"/> </details>  
    </td>
    <td>
      <img src="https://img.shields.io/badge/Train_Free-808080" height="20">
    </td>
    <td>EVA-CLIP+</br>RAFT+</br>VideoLLaMA2.1-7B-16F</td>
    <td>82.03</td>
    <td>64.04</td>
  </tr>
  <tr>
    <td>2025</td>
    <td>LPG
      <a href="https://openreview.net/forum?id=4ua4wyAQLm"><img src="https://img.shields.io/badge/OpenReview-rgba(0,0,0,0)" height="20"></a> 
    </td>
    <td><a href="https://github.com/AllenYLJiang/ Local-Patterns-Generalize-Better"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> </td>
    <td> <details> <summary></summary> <img src="img/meth/80-2025-LPG.png" width="1500"/> </details></td>
    <td><img src="https://img.shields.io/badge/Unsupervised-808080" height="20"></td>
    <td>YOLOv7+</br>BLIP-2(I+T)</td>
    <td>?</td>
    <td>?</td>
  </tr>
  <tr>
    <td>2025</td>
    <td>ProDisc-VAD <a href="http://arxiv.org/abs/2505.02179"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td>
      <a href="https://github.com/modadundun/ProDisc-VAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> 
      <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20">
    </td>
    <td> <details> <summary></summary> <img src="img/meth/81-2025-PRODISCVAD.png" width="1500"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>CLIP</td>
    <td>87.12</td>
    <td></td>
  </tr>
  <tr>
    <td>2025</td>
    <td> <img src="https://img.shields.io/badge/SUrvllnce_VQA_589k-rgb(128, 120, 12)" height="20"> <a href="http://arxiv.org/abs/2505.12589"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td>
      <a href="https://huggingface.co/datasets/fei213/SurveillanceVQA-589K"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=Hugging Face" height="20"></a> 
      <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20">
    </td>
    <td> 
      <!-- <details> <summary></summary> <img src="img/ds/2025-SURVEILLANCEVQA589K1.png" width="2000"/> </details> -->
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
    <tr>
    <td>2025</td>
    <td>PiVAD <a href="http://arxiv.org/abs/2505.13123"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td>
      <a href="https://github.com/snehashismajhi/PI-VAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> 
      <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20">
    </td>
    <td> <details> <summary></summary> <img src="img/meth/82-2025-PIVAD.png" width="1500"/> </details></td>
    <td>
      <!-- <img src="https://img.shields.io/badge/Teach-Stud-white" height="20"> </br> -->
      <img src="https://img.shields.io/badge/Weak-FFA500" height="20">
    </td>
    <td>(TRAIN)</br>YOLOV7-pose+</br>DepthAnythingV2+</br>SAM+RAFT+VifiCLIP</br>(INFER)</br>I3D</td>
    <td>90.33/77.77 </td>
    <td>85.37/85.79</td>
  </tr>
  <tr>
    <td>2025</td>
    <td>Flashback
      <a href="http://arxiv.org/abs/2505.15205"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> 
    </td>
    <td>
      <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20">
    </td>
    <td> 
      <img src="https://img.shields.io/badge/Open-00BFFF" height="20"> </br>
      <img src="https://img.shields.io/badge/Explainable-green" height="20">
      <details> <summary></summary> <img src="img/meth/83-2025-FLASHBACK.png" width="1500"/> </details>
    </td>
    <td>
      <img src="https://img.shields.io/badge/Train_Free-808080" height="20">
    </td>
    <td>PerceptionEncoder(I+T)</td>
    <td>87.29</td>
    <td>75.13</td>
  </tr>
    <tr>
    <td>2025</td>
    <td> LaAP
      <a href="http://arxiv.org/abs/2505.19022"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"> </br>
      <img src="https://img.shields.io/badge/UCFC/MSAD-HN-rgb(128, 120, 12)" height="20">
    </td>
    <td> 
      <a href="https://github.com/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>
    </td>
    <td> 
      <details> 
        <summary>metrics</summary> 
        <img src="img/meth/84-2025-LAAP.png" width="1500"/> 
        <img src="img/meth/84-2025-LAAP2.png" width="1500"/> 
      </details>
      <details> 
        <summary>results</summary> 
        <img src="img/meth/84-2025-LAAP-reslaap.png" width="1500"/> 
        <img src="img/meth/84-2025-LAAP-resfar.png" width="1500"/> 
      </details>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr> 
    <tr>
    <td>2025</td>
    <td> VAD-R1
      <a href="http://arxiv.org/abs/2505.19877"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20">
      <td> <a href="https://github.com/wbfwonderful/Vad-R1"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a></td>
    </td>
    <td> 
      <img src="https://img.shields.io/badge/Explainable-green" height="20">
      <details> 
      <summary></summary> <img src="img/meth/85-2025-VADR1.png" width="1500"/> 
      </details>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr> 
  <tr>
    <td>2025</td>
    <td> VAU-R1
      <a href=" http://arxiv.org/abs/2505.23504"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20">
    </td>
    <td> 
      <a href="https://github.com/GVCLab/VAU-R1"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>
    </td>
    <td> 
      <img src="https://img.shields.io/badge/Explainable-green" height="20">
      <details> <summary></summary> <img src="img/meth/86-2025-VAUR1.png" width="1500"/> </details>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr> 
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr> 

  <tr>
    <td>2025</td>
    <td> FEDVAD
      <!-- <a href=" http://arxiv.org/abs/2505.23504"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"> -->
    </td>
    <td> 
      <a href="https://github.com/wbfwonderful/Fed-WSVAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>
    </td>
    <td> 
      <img src="https://img.shields.io/badge/Federated Learning-pink" height="20">
      <details> <summary></summary> <img src="img/meth/FEDWSVAD.png" width="1500"/> </details>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr> 

  
  <!--
  <tr>
    <td>2024</td>
    <td> 
      <a href="http://arxiv.org/abs/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20">
      <td> <a href="https://github.com/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a></td>
    </td>
    <td> <details> <summary></summary> <img src="img/meth" width="1500"/> </details></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr> 
  -->
</tbody></table>





## HAWK
<table>
  <tr><th colspan="8"><strong>(A) Anomaly Video Description Generation</strong></th></tr>
  <tr>
    <th>Method</th>
    <th colspan="4">Text-Level </th>
    <th colspan="3">GPT-Guided </th>
  </tr>
  <tr>
    <td></td>
    <th>BLEU-1</th>
    <th>BLEU-2</th>
    <th>BLEU-3</th>
    <th>BLEU-4</th>
    <th>Reasonability</th>
    <th>Detail</th>
    <th>Consistency</th>
  </tr>
  <tr>
    <td>Video-ChatGPT [26]</td>
    <td>0.107</td>
    <td>0.046</td>
    <td>0.017</td>
    <td>0.008</td>
    <td>0.084</td>
    <td>0.108</td>
    <td>0.055</td>
  </tr>
  <tr>
    <td>VideoChat [15]</td>
    <td>0.053</td>
    <td>0.023</td>
    <td>0.008</td>
    <td>0.003</td>
    <td>0.107</td>
    <td>0.205</td>
    <td>0.054</td>
  </tr>
  <tr>
    <td>Video-LLaMA [46]</td>
    <td>0.062</td>
    <td>0.025</td>
    <td>0.009</td>
    <td>0.004</td>
    <td>0.120</td>
    <td>0.217</td>
    <td>0.066</td>
  </tr>
  <tr>
    <td>LLaMA-Adapter [47]</td>
    <td>0.132</td>
    <td>0.052</td>
    <td>0.018</td>
    <td>0.008</td>
    <td>0.060</td>
    <td>0.091</td>
    <td>0.038</td>
  </tr>
  <tr>
    <td>Video-LLaVA [17]</td>
    <td>0.071</td>
    <td>0.030</td>
    <td>0.012</td>
    <td>0.005</td>
    <td>0.077</td>
    <td>0.115</td>
    <td>0.038</td>
  </tr>
  <tr>
    <td>HAWK</td>
    <td>0.270</td>
    <td>0.139</td>
    <td>0.074</td>
    <td>0.043</td>
    <td>0.283</td>
    <td>0.320</td>
    <td>0.218</td>
  </tr>

  <tr><th colspan="8"><strong>(B) Anomaly Video Question-Answering</strong></th></tr>
  <tr>
    <th>Method</th>
    <th>BLEU-1</th>
    <th>BLEU-2</th>
    <th>BLEU-3</th>
    <th>BLEU-4</th>
    <th>Reasonability</th>
    <th>Detail</th>
    <th>Consistency</th>
  </tr>
  <tr>
    <td>Video-ChatGPT [26]</td>
    <td>0.177</td>
    <td>0.096</td>
    <td>0.058</td>
    <td>0.038</td>
    <td>0.508</td>
    <td>0.430</td>
    <td>0.421</td>
  </tr>
  <tr>
    <td>VideoChat [15]</td>
    <td>0.261</td>
    <td>0.133</td>
    <td>0.074</td>
    <td>0.043</td>
    <td>0.699</td>
    <td>0.631</td>
    <td>0.598</td>
  </tr>
  <tr>
    <td>Video-LLaMA [46]</td>
    <td>0.156</td>
    <td>0.081</td>
    <td>0.045</td>
    <td>0.027</td>
    <td>0.586</td>
    <td>0.485</td>
    <td>0.497</td>
  </tr>
  <tr>
    <td>LLaMA-Adapter [47]</td>
    <td>0.199</td>
    <td>0.109</td>
    <td>0.067</td>
    <td>0.043</td>
    <td>0.646</td>
    <td>0.559</td>
    <td>0.549</td>
  </tr>
  <tr>
    <td>Video-LLaVA [17]</td>
    <td>0.094</td>
    <td>0.054</td>
    <td>0.034</td>
    <td>0.023</td>
    <td>0.393</td>
    <td>0.274</td>
    <td>0.316</td>
  </tr>
  <tr>
    <td>HAWK</td>
    <td  style="font-weight:bold">0.319</td>
    <td  style="font-weight:bold">0.179</td>
    <td  style="font-weight:bold">0.112</td>
    <td  style="font-weight:bold">0.073</td>
    <td  style="font-weight:bold">0.840</td>
    <td  style="font-weight:bold">0.794</td>
    <td  style="font-weight:bold">0.753</td>
  </tr>
</table>


## HIVAU-70K
<table>
  <caption>Comparison of reasoning performance with state-of-the-art Multimodal Large Language Models (MLLMs). 'BLEU' refers to the cumulative values from BLEU-1 to BLEU-4. We evaluate the quality of the generated text at different granularities, including clip-level (C), event-level (E), and video-level (V).</caption>
  <tr>
    <th>Method</th>
    <th>Params</th>
    <th colspan="3">BLEU (↑)</th>
    <th colspan="3">CIDEr (↑)</th>
    <th colspan="3">METEOR (↑)</th>
    <th colspan="3">ROUGE (↑)</th>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <th>C</th>
    <th>E</th>
    <th>V</th>
    <th>C</th>
    <th>E</th>
    <th>V</th>
    <th>C</th>
    <th>E</th>
    <th>V</th>
    <th>C</th>
    <th>E</th>
    <th>V</th>
  </tr>
  <tr>
    <td>Video-ChatGPT</td>
    <td>7B</td>
    <td>0.152</td>
    <td>0.068</td>
    <td>0.066</td>
    <td>0.033</td>
    <td>0.011</td>
    <td>0.013</td>
    <td>0.102</td>
    <td>0.069</td>
    <td>0.044</td>
    <td>0.153</td>
    <td>0.048</td>
    <td>0.079</td>
  </tr>
  <tr>
    <td>Video-LLaMA</td>
    <td>7B</td>
    <td>0.151</td>
    <td>0.079</td>
    <td>0.104</td>
    <td>0.024</td>
    <td>0.014</td>
    <td>0.017</td>
    <td>0.112</td>
    <td>0.076</td>
    <td>0.057</td>
    <td>0.156</td>
    <td>0.067</td>
    <td>0.090</td>
  </tr>
  <tr>
    <td>Video-LLaVA</td>
    <td>7B</td>
    <td>0.164</td>
    <td>0.046</td>
    <td>0.055</td>
    <td>0.032</td>
    <td>0.009</td>
    <td>0.013</td>
    <td>0.097</td>
    <td>0.022</td>
    <td>0.014</td>
    <td>0.132</td>
    <td>0.023</td>
    <td>0.045</td>
  </tr>
  <tr>
    <td>LLaVA-Next-Video</td>
    <td>7B</td>
    <td>0.435</td>
    <td>0.091</td>
    <td>0.120</td>
    <td>0.102</td>
    <td>0.015</td>
    <td>0.031</td>
    <td>0.117</td>
    <td>0.085</td>
    <td>0.096</td>
    <td>0.198</td>
    <td>0.080</td>
    <td>0.106</td>
  </tr>
  <tr>
    <td>QwenVL2</td>
    <td>7B</td>
    <td>0.312</td>
    <td>0.101</td>
    <td>0.155</td>
    <td>0.044</td>
    <td>0.020</td>
    <td>0.044</td>
    <td>0.133</td>
    <td>0.092</td>
    <td>0.101</td>
    <td>0.163</td>
    <td>0.081</td>
    <td>0.137</td>
  </tr>
  <tr>
   <td>InternVL2</td>
    <td>8B</td>
    <td>0.331</td>
    <td>0.101</td>
    <td>0.145</td>
    <td>0.052</td>
    <td>0.022</td>
    <td>0.035</td>
    <td>0.141</td>
    <td>0.095</td>
    <td>0.101</td>
    <td>0.182</td>
    <td>0.102</td>
    <td>0.122</td>
  </tr>
  <tr>
    <td>Holmes-VAU</td>
    <td>2B</td>
    <td  style="font-weight:bold">0.913</td>
    <td  style="font-weight:bold">0.804</td>
    <td  style="font-weight:bold">0.566</td>
    <td  style="font-weight:bold">0.467</td>
    <td  style="font-weight:bold">1.519</td>
    <td  style="font-weight:bold">1.437</td>
    <td  style="font-weight:bold">0.190</td>
    <td  style="font-weight:bold">0.165</td>
    <td  style="font-weight:bold">0.121</td>
    <td  style="font-weight:bold">0.329</td>
    <td  style="font-weight:bold">0.370</td>
    <td  style="font-weight:bold">0.355</td>
  </tr>
</table>


## MSAD

<table>
  <caption>
    <em># : LaAP implemention</em></br>
    <em>* : π-VAD</em></br>
    <em>ä : MSAD</em>
  </caption>
  <tr>
    <th></th>
    <th>Feature</th>
    <th>AUC</th>
    <th>AUC<sub>A</sub></th>
    <th>AP</th>
    <th>AP<sub>A</sub></th>
  </tr>
  <tr>
    <td>RTFM</td>
    <td>
      <sup>#</sup></br>
      I3D<sup>ä</sup></br>
      VST<sup>ä</sup>
    </td>    
    <td>86.65</br>85.67</td>
    <td>-</br>-</br>-</td>
    <td>-</br>-</br>-</td>
    <td>-</br>-</br>-</td>
    <td>71.3</br>-</br>-</td>
  </tr>
  <tr>
    <td>MGFN</td>
    <td>
      <sup>#</sup></br>
      I3D<sup>ä</sup></br>
      VST<sup>ä</sup>
    </td>
    <td>84.96</br>78.94</td>
    <td>-</br>-</br>-</td>
    <td>-</br>-</br>-</td>
    <td>-</br>-</br>-</td>
    <td>66.4</br>-</br>-</td>
  </tr>
  <tr>
    <td>PEL4VAD</td>
    <td>
      <sup>#</sup>
    </td>
    <td>87.3</td>
    <td>-</td>
    <td>67.6</td>
    <td>-</td>
    <td>72.9</td>
  </tr>
  <tr>
    <td>VadCLIP</td>
    <td>
      <sup>#</sup>
    </td>
    <td>87.8</td>
    <td>-</td>
    <td>59.7</td>
    <td>-</td>
    <td>66.5</td>
  </tr>
  <tr>
    <td>TEVAD</td>
    <td>
      I3D<sup>ä</sup></br>
      VST<sup>ä</sup>
    </td>
    <td>86.82</br>83.60</td>
    <td>-</br>-</td>
    <td>-</br>-</td>
    <td>-</br>-</td>
    <td>-</br>-</td>
  </tr>
  <tr>
    <td>UR-DMU</td>
    <td>
      <sup>#</sup></br>
      I3D<sup>*</sup></br>
      I3D<sup>ä</sup></br>
      VST<sup>ä</sup></td>
    <td>87.8</br>85.78</br>85.02</br>72.36</td>
    <td>-</br>67.95</br>-</br>-</td>
    <td>69.7</br>67.35</br>-</br>-</td>
    <td>-</br>75.30</br>-</br>-</td>
    <td>72.9</br>-</br>-</br>-</td>
  </tr>
  <tr>
    <td>π-VAD</td>
    <td>I3D</td>
    <td><strong>88.68</strong></td>
    <td><strong>71.25</strong></td>
    <td><strong>71.26</strong></td>
    <td><strong>77.86</strong></td>
    <td>-</td>
  </tr>
</table>



## ECVA


## UCFC-DVS


## SurveillanceVQA-589K


## VAD-Reasoning

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Params.</th>
      <th colspan="3">Anomaly Reasoning</th>
      <th colspan="5">Anomaly Detection</th>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <th>BLEU-2</th>
      <th>METEOR</th>
      <th>ROUGE-2</th>
      <th>Acc</th>
      <th>F1</th>
      <th>mIoU</th>
      <th>R@0.3</th>
      <th>R@0.5</th>
    </tr>
  </thead>
  <tbody>
    <!-- Group Headers -->
    <tr><td colspan="10" style=" font-weight:bold;">Open-Source video MLLMs</td></tr>
    <tr>
      <td>InternVideo2.5 </td>
      <td>8B</td>
      <td>0.110</td>
      <td>0.264</td>
      <td>0.109</td>
      <td>0.715</td>
      <td>0.730</td>
      <td>0.417</td>
      <td>0.458</td>
      <td>0.424</td>
    </tr>
    <tr>
      <td>InternVL3 </td>
      <td>8B</td>
      <td>0.124</td>
      <td>0.286</td>
      <td>0.116</td>
      <td>0.779</td>
      <td>0.756</td>
      <td>0.550</td>
      <td>0.613</td>
      <td>0.540</td>
    </tr>
    <tr>
      <td>VideoChat-Flash </td>
      <td>7B</td>
      <td>0.012</td>
      <td>0.084</td>
      <td>0.047</td>
      <td>0.683</td>
      <td>0.487</td>
      <td>0.536</td>
      <td>0.538</td>
      <td>0.358</td>
    </tr>
    <tr>
      <td>VideoLLaMA3 </td>
      <td>7B</td>
      <td>0.066</td>
      <td>0.200</td>
      <td>0.092</td>
      <td>0.665</td>
      <td>0.624</td>
      <td>0.425</td>
      <td>0.451</td>
      <td>0.419</td>
    </tr>
    <tr>
      <td>LLaVA-NeXT-Video </td>
      <td>7B</td>
      <td>0.113</td>
      <td>0.264</td>
      <td>0.116</td>
      <td>0.761</td>
      <td>0.730</td>
      <td>0.567</td>
      <td>0.610</td>
      <td>0.563</td>
    </tr>
    <tr><td colspan="10" style=" font-weight:bold;">Open-Source video reasoning MLLMs</td></tr>
    <tr>
      <td>Open-R1-Video</td>
      <td>7B</td>
      <td>0.060</td>
      <td>0.179</td>
      <td>0.084</td>
      <td>0.793</td>
      <td>0.790</td>
      <td>0.559</td>
      <td>0.642</td>
      <td>0.540</td>
    </tr>
    <tr>
      <td>Video-R1 </td>
      <td>7B</td>
      <td>0.135</td>
      <td>0.317</td>
      <td>0.132</td>
      <td>0.624</td>
      <td>0.694</td>
      <td>0.334</td>
      <td>0.392</td>
      <td>0.328</td>
    </tr>
    <tr>
      <td>VideoChat-R1 </td>
      <td>7B</td>
      <td>0.128</td>
      <td>0.287</td>
      <td>0.123</td>
      <td>0.793</td>
      <td>0.790</td>
      <td>0.559</td>
      <td>0.642</td>
      <td>0.540</td>
    </tr>
    <tr><td colspan="10" style=" font-weight:bold;">MLLM-based VAD methods</td></tr>
    <tr>
      <td>Holmes-VAD </td>
      <td>7B</td>
      <td>0.003</td>
      <td>0.074</td>
      <td>0.027</td>
      <td>0.565</td>
      <td>0.120</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Holmes-VAU </td>
      <td>2B</td>
      <td>0.077</td>
      <td>0.182</td>
      <td>0.075</td>
      <td>0.490</td>
      <td>0.371</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td >HAWK </td>
      <td>7B</td>
      <td>0.042</td>
      <td>0.156</td>
      <td>0.042</td>
      <td>0.513</td>
      <td>0.648</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr><td colspan="10" style=" font-weight:bold;">Proprietary MLLMs</td></tr>
    <tr>
      <td>Claude3.5-Haiku </td>
      <td>-</td>
      <td>0.097</td>
      <td>0.253</td>
      <td>0.098</td>
      <td>0.580</td>
      <td>0.354</td>
      <td>0.518</td>
      <td>0.543</td>
      <td>0.524</td>
    </tr>
    <tr>
      <td>GPT-4o </td>
      <td>-</td>
      <td>0.154</td>
      <td>0.341</td>
      <td>0.133</td>
      <td>0.711</td>
      <td>0.760</td>
      <td>0.472</td>
      <td>0.565</td>
      <td>0.476</td>
    </tr>
    <tr>
      <td>Gemini2.5-Flash </td>
      <td>-</td>
      <td>0.133</td>
      <td>0.308</td>
      <td>0.120</td>
      <td>0.624</td>
      <td>0.707</td>
      <td>0.370</td>
      <td>0.437</td>
      <td>0.358</td>
    </tr>
    <tr><td colspan="10" style="font-weight:bold;">Proprietary reasoning MLLMs</td></tr>
    <tr>
      <td>Gemini2.5-pro </td>
      <td>-</td>
      <td>0.145</td>
      <td>0.356</td>
      <td>0.137</td>
      <td>0.829</td>
      <td>0.836</td>
      <td>0.636</td>
      <td>0.722</td>
      <td>0.638</td>
    </tr>
    <tr>
      <td>QVQ-Max </td>
      <td>-</td>
      <td>0.142</td>
      <td>0.318</td>
      <td>0.121</td>
      <td>0.702</td>
      <td>0.747</td>
      <td>0.430</td>
      <td>0.503</td>
      <td>0.412</td>
    </tr>
    <tr>
      <td>o4-mini </td>
      <td>-</td>
      <td>0.106</td>
      <td>0.263</td>
      <td>0.109</td>
      <td style="font-weight:bold">0.884</td>
      <td style="font-weight:bold">0.875</td>
      <td>0.644</td>
      <td>0.736</td>
      <td>0.631</td>
    </tr>
    <tr>
      <td style="font-weight:bold;">Vad-R1 </td>
      <td>7B</td>
      <td style="font-weight:bold">0.233</td>
      <td style="font-weight:bold">0.406</td>
      <td style="font-weight:bold">0.194</td>
      <td>0.875</td>
      <td>0.862</td>
      <td style="font-weight:bold">0.713</td>
      <td style="font-weight:bold">0.770</td>
      <td style="font-weight:bold">0.706</td>
    </tr>
  </tbody>
</table>

# Other Collections/Resources

- Generalized Video Anomaly Event Detection: Systematic Taxonomy and Comparison of Deep Models [![](https://img.shields.io/badge/-9B59B6?logo=github&logoColor=white)](https://github.com/fudanyliu/GVAED) -> 

- Networking Systems for Video Anomaly Detection: A Tutorial and Survey [![](https://img.shields.io/badge/-9B59B6?logo=github&logoColor=white)](https://github.com/fdjingliu/NSVAD) -> A great starting point for VAD, covering supervision development and progression (U,Ws,Fu)

- [Quo Vadis, Anomaly Detection?  LLMs and VLMs in the Spotlight](http://arxiv.org/abs/2412.18298) [![](https://img.shields.io/badge/-black?logo=github&logoColor=white)](https://github.com/Darcyddx/VAD-LLM) -> A survey the integration of large language models (LLMs) and vision-language models (VLMs) in video anomaly detection (VAD)

- [Markdown-Cheatsheet](https://github.com/lifeparticle/Markdown-Cheatsheet)




<!---

HTTPS


ICONS
  https://simpleicons.org/
  https://shields.io/docs/logos   https://badges.pages.dev/
  

  https://img.shields.io/badge/any_text-you_like-blue
  https://img.shields.io/badge/just%20the%20message-8A2BE2
-->



<!-- 
| Year | Method |  Feature & Supervision | UCF (AUCo,AUCa,FAR) | XDV (APo,APa,FAR) |
|------|--------|------------------------|---------------------|-------------------|
| 2018 | MIR  [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/Roc-Ng/DeepMIL) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) ![](https://img.shields.io/badge/uws4vad-purple) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) | 0.7541 <br> 0.7792 |  <br>  | 
| 2019 | GCN [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/jx-zhong-for-academic-purpose/GCN-Anomaly-Detection) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> TSN ![](https://img.shields.io/badge/W-FFA500) | 0.8192 <br> 0.8212 |  <br>  |
| 2019 | MA [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | PWC-OF ![](https://img.shields.io/badge/W-FFA500) | 0.7210 |  |
| 2019 | TCN [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) | 0.7866 |  |
| 2020 | SRG [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) | 0.7954/-/0.13 |  |
| 2020 | ARN [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.7571 |  |
| 2020 | HLN/XDV [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/Roc-Ng/XDVioDet) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | 0.8244 <br>  |  <br> 0.7864 |
| 2020 | WSAL [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/ktr-hubrt/WSAL) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) <br> R(2+1)D ![](https://img.shields.io/badge/W-FFA500) <br> TSN ![](https://img.shields.io/badge/W-FFA500) | 0.7418 <br> 0.7529 <br> 0.8538,0.6738/- |  <br>  <br>  |
| 2020 | CLAWS [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) | 0.8303/-/0.12 |  |
| 2021 | MIST [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8140/-/2.19 <br> 0.8230/-/0.13 |  <br>  |
| 2021 | AVF [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D+VGG ![](https://img.shields.io/badge/W-FFA500) |  | 0.8169 |
| 2021 | RTFM [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/tianyu0207/RTFM) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) ![](https://img.shields.io/badge/uws4vad-purple) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8328 <br> 0.8430 | 0.7589 <br> 0.7781 |
| 2021 | XEL [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/sdjsngs/XEL-WSAD) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) | 0.8260 |  |
| 2021 | CA [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/changsn/Contrastive-Attention-for-Video-Anomaly-Detection) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> TSN <br> I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8340 <br> 0.8352 <br> 0.8462 |  <br>  <br> 0.7690 |
| 2021 | MS-BS [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8353 |  |
| 2021 | DAM [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=tensorflow)](https://github.com/snehashismajhi/DAM-Anomaly-Detection) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8267/-/0.3 |  |
| 2022 | CMALA [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/yujiangpu20/cma_xdVioDet) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) ![](https://img.shields.io/badge/uws4vad-purple) | I3D+VGG ![](https://img.shields.io/badge/W-FFA500) |  | 0.8354 |
| 2022 | CLAWS+ [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> 3DRN ![](https://img.shields.io/badge/W-FFA500) | 0.8337/-/0.11 <br> 0.8416/-/0.09 |  <br>  |
| 2022 | WSTR [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/justsmart/WSTD-VAD) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8317 |  |
| 2022 | STA [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8160 <br> 0.8300 |  <br>  |
| 2022 | MSL [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/xidianai/MSL) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) <br> VSWIN | 0.8285 <br> 0.8530 <br> 0.8562 | 0.7553 <br> 0.7828 <br> 0.7859 |
| 2022 | SGMIR [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8170 |  |
| 2022 | TCA [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8208/-/0.11 <br> 0.8375/-/0.05 |  <br>  |
| 2022 | MACIL [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/JustinYuu/MACIL_SD) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) ![](https://img.shields.io/badge/uws4vad-purple) | I3D+VGG ![](https://img.shields.io/badge/W-FFA500) |  | 0.8340 |
| 2022 | LAN [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8512 | 0.8072 |
| 2022 | ANM [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/sakurada-cnq/salient_feature_anomaly) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) ![](https://img.shields.io/badge/uws4vad-purple) | I3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | 0.8299 <br>  |  <br> 0.8491 |
| 2022 | MSAF [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/Video-AD/MSFA) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D+OF ![](https://img.shields.io/badge/W-FFA500) <br> I3D+OF+VGG ![](https://img.shields.io/badge/W-FFA500) | 0.8134 <br>  |  <br> 0.8051 |
| 2022 | TAI [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8573 |  |
| 2022 | BSME [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8363 |  |
| 2022 | MGFN [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/carolchenyx/MGFN) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) ![](https://img.shields.io/badge/uws4vad-purple) | I3D ![](https://img.shields.io/badge/W-FFA500) <br> VSwin ![](https://img.shields.io/badge/W-FFA500) | 0.8698 <br> 0.8667 | 0.7919 <br> 0.8011 |
| 2022 | CUN [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/ArielZc/CU-Net) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | 0.8622 <br>  | 0.7874 <br> 0.8143 |
| 2022 | CLIP-TSA [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/joos2010kj/CLIP-TSA) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) ![](https://img.shields.io/badge/uws4vad-purple) | CLIP ![](https://img.shields.io/badge/W-FFA500) | 0.8758 | 0.8219 |
| 2023 | NGMIL [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8343 <br> 0.8563 | 0.7591 <br> 0.7851 |
| 2023 | URDMU [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/henrryzh1/UR-DMU) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | 0.8697/-/1.05 <br>  | 0.8166/-/0.65 <br> 0.8177 |
| 2023 | UMIL [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/ktr-hubrt/UMIL) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](https://arxiv.org/pdf/2303.12369v1) | X-CLIP ![](https://img.shields.io/badge/W-FFA500) | 0.8675,0.6868/- |  |
| 2023 | LSTC [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/shengyangsun/LSTC_VAD) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8347 <br> 0.8588 |  <br>  |
| 2023 | BERTMIL [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/wjtan99/BERT_Anomaly_Video_Classification) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D+FLOW ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8671 <br>  |  <br> 0.8210 |
| 2023 | SLAMBS [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8619 | 0.8423 |
| 2023 | TEVAD [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/coranholmes/TEVAD) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | | | |
| 2023 | HYPERVD [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/xiaogangpeng/HyperVD) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D+VGG ![](https://img.shields.io/badge/W-FFA500) |  | 0.8567 |
| 2023 | PEL4VAD [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/yujiangpu20/PEL4VAD) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8676,0.7224,0.43 | 0.8559,0.7026,0.57 |
| 2023 | VAR [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/Roc-Ng/VAR) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | | | |
| 2023 | CNN-VIT [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) <br> CLIP ![](https://img.shields.io/badge/W-FFA500) <br> C3D+CLIP ![](https://img.shields.io/badge/W-FFA500) <br> I3D+CLIP ![](https://img.shields.io/badge/W-FFA500) | 0.8578 <br> 0.8650 <br> 0.8763 <br> 0.8802 <br> 0.8897 |  <br>  <br>  <br>  <br>  |
| 2023 | UCFA [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | | | |
| 2023 | SAA [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/2023-MindSpore-4/Code4/tree/main/WS-VAD-mindspore-main) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | 0.8619,0.6877/- <br>  | 0.8359,0.8419/- <br> 0.8423 |
| 2023 | ANOMCLIP [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/lucazanella/AnomalyCLIP) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | ViT-B/16 ![](https://img.shields.io/badge/W-FFA500) | 0.8636 | 0.7851 |
| 2023 | MTDA [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D+VGG ![](https://img.shields.io/badge/W-FFA500) |  | 0.8444 |
| 2023 | BNWVAD [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/cool-xuan/BN-WVAD) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) ![](https://img.shields.io/badge/uws4vad-purple) | I3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | 0.8724,0.7171/- <br>  | 0.8493,0.8545/- <br> 0.8526 |
| 2023 | DEN [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/ArielZc/DE-Net) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | 0.8633 <br>  | 0.8166 <br> 0.8313 |
| 2023 | VADCLIP [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/nwpu-zxr/VadCLIP) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D+CLIP ![](https://img.shields.io/badge/W-FFA500) | 0.8802 | 0.8451 |
| 2024 | VAD-LLaAMa [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/ktr-hubrt/VAD-LLaMA) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](http://arxiv.org/abs/2401.05702) | | | |
| 2024 | GlanceVAD [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/pipixin321/GlanceVAD) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](http://arxiv.org/abs/2403.06154) ![](https://img.shields.io/badge/uws4vad-purple) | | | |
| 2024 | LAVAD [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/lucazanella/lavad) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](http://arxiv.org/abs/2404.01014) | | | |
| 2024 | MSBT [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/shengyangsun/MSBT) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](http://arxiv.org/abs/2405.05130) | I3D+VGG ![](https://img.shields.io/badge/W-FFA500) <br> I3D+VGG+TV-L1 ![](https://img.shields.io/badge/W-FFA500)|  <br>  | 0.8254 <br> 0.8432 |
| 2024 | HAWK [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/jqtangust/hawk) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](http://arxiv.org/abs/2405.16886) | | | |
| 2024 | PEMIL [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/Junxi-Chen/PE-MIL) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](https://ieeexplore.ieee.org/document/10657732/) | | | |
| 2024 | Holmes-VAD [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/pipixin321/HolmesVAD)[![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](http://arxiv.org/abs/2406.12235) | | | |
| 2024 | Holmes-VAU [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/pipixin321/HolmesVAU)[![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2412.06171) | | | |
||||||
-->