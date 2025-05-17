
Below are the collected works (mainly) under Weakly supervision (UCFC/XDV), but also new benchamrks.

Results refer to the test subset which methods were evaluated, *o(verall)/a(nomaly)*, when provided.

<table><thead>
  <tr>
    <th>Year</th>
    <th>Method,Code,Paper</th>
    <th>Anom Criterion</th>
    <th>Supervision</th>
    <th>Feature</th>
    <th>UCF <br>(AUCo,AUCa,FAR)</th>
    <th>XDV <br>(APo,APa,FAR)</th>
  </tr></thead>
<tbody>
  <tr>
    <td>2018</td>
    <td>MIR <a href=""><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="#"> <img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--green?labelColor=purple" height="20"></td>
    <td> <details> <summary></summary> <img src="./img/meth/0-2018-MIR.png"/> </details>  </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D<br>I3D</td>
    <td>0.7541,-,-<br>0.7792,-,-</td>
    <td>-,-,-<br>-,-,-</td>
  </tr>
  <tr>
    <td>2019</td>
    <td>GCN <a href="https://github.com/jx-zhong-for-academic-purpose/GCN-Anomaly-Detection"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://ieeexplore.ieee.org/abstract/document/8953791"> <img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D<br>TSN</td>
    <td>0.8192,-,-<br>0.8212,-,-</td>
    <td>-,-,-<br>-,-,-</td>
  </tr>
  <tr>
    <td>2019</td>
    <td>MA  <a href="#"> <img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>PWC-OF</td>
    <td>0.7210,-,-</td>
    <td>-,-,-</td>
  </tr>
  <tr>
    <td>2019</td>
    <td>TCN  <a href="#"> <img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D </td>
    <td>0.7866,-,-</td>
    <td>-,-,-</td>
  </tr>
  <tr>
    <td>2020</td>
    <td>SRG  <a href="#"> <img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D </td>
    <td>0.7954,-,0.13</td>
    <td>-,-,-</td>
  </tr>
  <tr>
    <td>2020</td>
    <td>ARN <a href="https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2104.07268"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D </td>
    <td>0.7571,-,-</td>
    <td>-,-,-</td>
  </tr>
  <tr>
    <td>2020</td>
    <td>HLN/XDV <a href="https://github.com/Roc-Ng/XDVioDet"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://roc-ng.github.io/XD-Violence/images/paper.pdf"><img src="https://img.shields.io/badge/ECCV-rgba(0,0,0,0)" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D<br>I3D+VGG</td>
    <td>0.8244,-,-<br>-,-,-</td>
    <td>-,-,-<br>0.7864,-,-</td>
  </tr>
  <tr>
    <td>2020</td>
    <td>WSAL <a href="https://github.com/ktr-hubrt/WSAL"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://ieeexplore.ieee.org/abstract/document/9408419"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D <br>R(2+1)D<br>TSN</td>
    <td>0.7418,-,- <br> 0.7529,-,- <br> 0.8538,0.6738,-</td>
    <td>-,-,- <br> -,-,- <br> -,-,-</td>
  </tr>
  <tr>
    <td>2020</td>
    <td>CLAWS <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D</td>
    <td>0.8303,-,0.12</td>
    <td>-,-,-</td>
  </tr>
  <tr>
    <td>2021</td>
    <td>MIST <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D <br>I3D</td>
    <td>0.8140,-,2.19 <br> 0.8230,-,0.13</td>
    <td>-,-,- <br> -,-,-</td>
  </tr>
  <tr>
    <td>2021</td>
    <td>AVF <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D+VGG</td>
    <td>-,-,-</td>
    <td>0.8169,-,-</td>
  </tr>
  <tr>
    <td>2021</td>
    <td>RTFM <a href="https://github.com/tianyu0207/RTFM"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2101.10030"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--green?labelColor=purple" height="20"></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D <br>I3D</td>
    <td>0.8328,-,- <br> 0.8430,-,-</td>
    <td>0.7589,-,- <br> 0.7781,-,-</td>
  </tr>
  <tr>
    <td>2021</td>
    <td>XEL <a href="https://github.com/sdjsngs/XEL-WSAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://ieeexplore.ieee.org/document/9560033/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--yellow?labelColor=purple" height="20"> </td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D</td>
    <td>0.8260,-,-</td>
    <td>-,-,-</td>
  </tr>
  <tr>
    <td>2021</td>
    <td>CA <a href="https://github.com/changsn/Contrastive-Attention-for-Video-Anomaly-Detection"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://ieeexplore.ieee.org/document/9540293"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D <br>TSN <br>I3D </td>
    <td>0.8340,-,- <br> 0.8352,-,- <br> 0.8462,-,-</td>
    <td>-,-,- <br> -,-,- <br> 0.7690,-,-</td>
  </tr>
  <tr>
    <td>2021</td>
    <td>MS-BS  <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D</td>
    <td>0.8353,-,-</td>
    <td>-,-,-</td>
  </tr>
  <tr>
    <td>2021</td> 
    <td>DAM <a href="https://github.com/snehashismajhi/DAM-Anomaly-Detection"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=tensorflow" height="20"></a> <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D</td>
    <td>0.8267,-,0.3</td>
    <td>-,-,-</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>CMALA <a href="https://github.com/yujiangpu20/cma_xdVioDet"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://ieeexplore.ieee.org/document/9712793"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--green?labelColor=purple" height="20"></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D+VGG</td>
    <td>-,-,-</td>
    <td>0.8354,-,-</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>CLAWS+ <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D <br>3DRN </td>
    <td>0.8337,-,0.11 <br> 0.8416,-,0.09</td>
    <td>-,-,- <br> -,-,-</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>WSTR <a href="https://github.com/justsmart/WSTD-VAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://ieeexplore.ieee.org/abstract/document/9774889"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D </td>
    <td>0.8317,-,-</td>
    <td>-,-,-</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>STA <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D<br>I3D</td>
    <td>0.8160,-,- <br> 0.8300,-,-</td>
    <td>-,-,- <br> -,-,-</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>MSL <a href="https://github.com/LiShuo1001/MSL"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D <br>I3D <br>VSWIN</td>
    <td>0.8285,-,- <br> 0.8530,-,- <br> 0.8562,-,-</td>
    <td>0.7553,-,- <br> 0.7828,-,- <br> 0.7859,-,-</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>SGMIR <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D</td>
    <td>0.8170,-,-</td>
    <td>-,-,-</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>TCA <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D <br>I3D</td>
    <td>0.8208,-,0.11 <br> 0.8375,-,0.05</td>
    <td>-,-,- <br> -,-,-</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>MACIL <a href="https://github.com/JustinYuu/MACIL_SD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2207.05500"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--yellow?labelColor=purple" height="20"></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D+VGG </td>
    <td>-,-,-</td>
    <td>0.8340,-,-</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>LAN <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D </td>
    <td>0.8512,-,-</td>
    <td>0.8072,-,-</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>OpenVAD <a href="https://github.com/YUZ128pitt/Towards-OpenVAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://arxiv.org/abs/2208.11113"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td></td>
    <td><img src="https://img.shields.io/badge/OpenWorld-00BFFF" height="20"></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2022</td>
    <td>ANM <a href="https://github.com/sakurada-cnq/salient_feature_anomaly"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2209.06435"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--green?labelColor=purple" height="20"></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D <br>I3D+VGG</td>
    <td>0.8299,-,- <br> -,-,-</td>
    <td>-,-,- <br> 0.8491,-,-</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>MSAF <a href="https://github.com/Video-AD/MSFA"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D+OF <br>I3D+OF+VGG</td>
    <td>0.8134,-,- <br> -,-,-</td>
    <td>-,-,- <br> 0.8051,-,-</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>TAI <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D</td>
    <td>0.8573,-,-</td>
    <td>-,-,-</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>BSME <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D</td>
    <td>0.8363,-,-</td>
    <td>-,-,-</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>MGFN <a href="https://github.com/carolchenyx/MGFN."><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://arxiv.org/abs/2211.15098"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--yellow?labelColor=purple" height="20"></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D <br>VSwin</td>
    <td>0.8698,-,- <br> 0.8667,-,-</td>
    <td>0.7919,-,- <br> 0.8011,-,-</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>CUN <a href="https://github.com/ArielZc/CU-Net"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Exploiting_Completeness_and_Uncertainty_of_Pseudo_Labels_for_Weakly_Supervised_CVPR_2023_paper.pdf"><img src="https://img.shields.io/badge/CVPR-rgba(0,0,0,0)" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D <br>I3D+VGG </td>
    <td>0.8622,-,- <br> -,-,-</td>
    <td>0.7874,-,- <br> 0.8143,-,-</td>
  </tr>
  <tr>
    <td>2022</td>
    <td>CLIP-TSA <a href="https://github.com/joos2010kj/CLIP-TSA"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://arxiv.org/abs/2212.05136"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--yellow?labelColor=purple" height="20"></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>CLIP</td>
    <td>0.8758,-,-</td>
    <td>0.8219,-,-</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>NGMIL <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D <br>I3D </td>
    <td>0.8343,-,- <br> 0.8563,-,-</td>
    <td>0.7591,-,- <br> 0.7851,-,-</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>URDMU <a href="https://github.com/henrryzh1/UR-DMU"><i66mg src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2302.05160"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20"></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D <br>I3D+VGG</td>
    <td>0.8697,-,1.05 <br> -,-,-</td>
    <td>0.8166,-,0.65 <br> 0.8177,-,-</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>UMIL <a href="https://github.com/ktr-hubrt/UMIL"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2303.12369"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>X-CLIP</td>
    <td>0.8675,0.6868,-</td>
    <td>-,-,-</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>LSTC <a href="https://github.com/shengyangsun/LSTC_VAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D<br>I3D </td>
    <td>0.8347,-,- <br> 0.8588,-,-</td>
    <td>-,-,- <br> -,-,-</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>BERTMIL <a href="https://github.com/wjtan99/BERT_Anomaly_Video_Classification"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://arxiv.org/abs/2210.06688"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D+FLOW <br>I3D </td>
    <td>0.8671,-,- <br> -,-,-</td>
    <td>-,-,- <br> 0.8210,-,-</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>SLAMBS <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D </td>
    <td>0.8619,-,-</td>
    <td>0.8423,-,-</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>TEVAD <a href="https://github.com/coranholmes/TEVAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2023</td>
    <td>HYPERVD <a href="https://github.com/xiaogangpeng/HyperVD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2305.18797"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D+VGG </td>
    <td>-,-,-</td>
    <td>0.8567,-,-</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>PEL4VAD <a href="https://github.com/yujiangpu20/PEL4VAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2306.14451"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20"> </td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D </td>
    <td>0.8676,0.7224,0.43</td>
    <td>0.8559,0.7026,0.57</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>VAR <a href="https://github.com/Roc-Ng/VAR"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2307.12545"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2023</td>
    <td>CNN-VIT <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>C3D <br>I3D <br>CLIP <br>C3D+CLIP&nbsp;&nbsp;<br>I3D+CLIP </td>
    <td>0.8578,-,- <br> 0.8650,-,- <br> 0.8763,-,- <br> 0.8802,-,- <br> 0.8897,-,-</td>
    <td>-,-,- <br> -,-,- <br> -,-,- <br> -,-,- <br> -,-,-</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>SAA <a href="https://github.com/YukiFan/vad-weakly"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://github.com/Daniel00008/WS-VAD-mindspore"><img src="https://img.shields.io/badge/MS-rgba(0,0,0,0)" height="20"></a> <a href="https://github.com/2023-MindSpore-4/Code4"><img src="https://img.shields.io/badge/MS-rgba(0,0,0,0)" height="20"></a> <a href="http://arxiv.org/abs/2309.16309"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--yellow?labelColor=purple" height="20"> </td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D <br>I3D+VGG </td>
    <td>0.8619,0.6877,- <br> -,-,-</td>
    <td>0.8359,0.8419,- <br> 0.8423,-,-</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>AnomCLIP <a href="https://github.com/lucazanella/AnomalyCLIP"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2310.02835"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>ViT-B/16 </td>
    <td>0.8636,-,-</td>
    <td>0.7851,-,-</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>MTDA <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D+VGG </td>
    <td>-,-,-</td>
    <td>0.8444,-,-</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>BNWVAD <a href="https://github.com/cool-xuan/BN-WVAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2311.15367"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--yellow?labelColor=purple" height="20"></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D <br>I3D+VGG </td>
    <td>0.8724,0.7171,- <br> -,-,-</td>
    <td>0.8493,0.8545,- <br> 0.8526,-,-</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>DEN <a href="https://github.com/ArielZc/DE-Net"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2312.01764"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20"> </td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D <br>I3D+VGG </td>
    <td>0.8633,-,- <br> -,-,-</td>
    <td>0.8166,-,- <br> 0.8313,-,-</td>
  </tr>
  <tr>
    <td>2023</td>
    <td>VADCLIP <a href="https://github.com/nwpu-zxr/VadCLIP"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2308.11681"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20"> </td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D+CLIP</td>
    <td>0.8802,-,-</td>
    <td>0.8451,-,-</td>
  </tr>
  <tr>
    <td>2024</td>
    <td>VAD-LLaMA <a href="https://github.com/ktr-hubrt/VAD-LLaMA"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2401.05702"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2024</td>
    <td>LAP <a href="http://arxiv.org/abs/2403.01169"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2024</td>
    <td>GlanceVAD <a href="https://github.com/pipixin321/GlanceVAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2403.06154"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--green?labelColor=purple" height="20"></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
    <tr>
    <td>2024 </td>
    <td>OVVAD <a href="http://arxiv.org/abs/2311.07042"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Open-00BFFF" height="20"> </td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2024</td>
    <td>LAVAD <a href="https://github.com/lucazanella/lavad"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://arxiv.org/abs/2404.01014"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td><img src="https://img.shields.io/badge/Explainable-green" height="20"></td>
    <td><img src="https://img.shields.io/badge/Train_Free-808080" height="20"></td>
    <td> BLIP-2 ensemble <br> Llama-2-13b-chat <br> ImageBind multimodal encoders </td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2024</td>
    <td>TPWNG <a href="http://arxiv.org/abs/2404.08531"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2024</td>
    <td>MSBT <a href="https://github.com/shengyangsun/MSBT"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://arxiv.org/abs/2405.05130"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>I3D+VGG <br>I3D+VGG+TV-L1 </td>
    <td>-,-,- <br> -,-,-</td>
    <td>0.8254,-,- <br> 0.8432,-,-</td>
  </tr>
  <tr>
    <td>2024</td>
    <td>HAWK <a href="https://github.com/jqtangust/hawk"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2024</td>
    <td>PEMIL <a href="https://github.com/Junxi-Chen/PE-MIL"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://ieeexplore.ieee.org/document/10657732/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2024</td>
    <td>VAD-LLaMA <a href="https://github.com/ktr-hubrt/VAD-LLaMA"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2401.05702"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr> 
  
  <tr>
    <td>2024</td>
    <td>UCFA-VALU <a href="https://github.com/Xuange923/Surveillance-Video-Understanding"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://ieeexplore.ieee.org/document/10656129/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a></td>
    <td> <img src="https://img.shields.io/badge/Dataset/Task-rgba(0,0,0,0)" height="20"> <img src="https://img.shields.io/badge/VideoCaption-rgba(0,0,0,0)" height="20"> </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr> 
  <tr>
    <td>2024</td>
    <td>Holmes-VAD <a href="https://github.com/pipixin321/HolmesVAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2406.12235"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>  
    <td>2024</td>
    <td>Holmes-VAU <a href="https://github.com/pipixin321/HolmesVAU"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2412.06171"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20"></td>
    <td><img src="https://img.shields.io/badge/Explainable-green" height="20"><img src="https://img.shields.io/badge/FineTune-LoRa-rgba(0,0,0,0)?labelColor=rgb(21,%20197,%20197)" height="20"> <img src="https://img.shields.io/badge/HIVAU70k-rgba(0,0,0,0)" height="20"> </td>
    <td></td>
    <td>InternVL2-2B (RGB+TEXT)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2024</td>
    <td>FE-VAD <a href="https://ieeexplore.ieee.org/document/10688326/"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">2025</td>
    <td rowspan="2">PLOVAD <a href="https://github.com/ctX-u/PLOVAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="https://ieeexplore.ieee.org/document/10836858"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=ieee&logoColor=blue" height="20"></a> </td>
    <td rowspan="2"> <img src="https://img.shields.io/badge/PromptTune-Categorization808080" height="20"> </td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td rowspan="2">CLIP</td>
    <td></td>
    <td></td>
  </tr>
    <td><img src="https://img.shields.io/badge/Open-00BFFF" height="20"></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2025</td>
    <td>MTFL </a> <a href="https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13517/3055069/MTFL--multi-timescale-feature-learning-for-weakly-supervised-anomaly/10.1117/12.3055069.full"><img src="https://img.shields.io/badge/ICMV-rgba(0,0,0,0)" height="20"></a> </td>
    <td>ac</td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>feat</td>
    <td>ucfc</td>
    <td>xdv</td>
  </tr>
  <tr>
    <td>2025</td>
    <td>Sherlock <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2502.18863"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20"></td>
    <td><img src="https://img.shields.io/badge/Explainable-green" height="20"></td>
    <td></td>
    <td>feat</td>
    <td>ucfc</td>
    <td>xdv</td>
  </tr>
  <tr>
    <td>2025</td>
    <td>UCFC-DVS <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2503.12905"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td>ac</td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>feat</td>
    <td>ucfc</td>
    <td>xdv</td>
  </tr>
  <!-- <tr>
    <td>2025</td>
    <td>VADMamba <a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2503.21169"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20"></td>
    <td>ac</td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>feat</td>
    <td>ucfc</td>
    <td>xdv</td>
  </tr> -->
  <tr>
    <td>2025</td>
    <td>VERA <a href="https://github.com/vera-framework/VERA"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2412.01095"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td><img src="https://img.shields.io/badge/Explainable-green" height="20"></td>
    <td></td>
    <td>feat</td>
    <td>ucfc</td>
    <td>xdv</td>
  </tr>
  <tr>
    <td>2025</td>
    <td>AVadCLIP <a href="http://arxiv.org/abs/2504.04495"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a></td>
    <td>ac</td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>feat</td>
    <td>ucfc</td>
    <td>xdv</td>
  </tr>
  <tr>
    <td>2025</td>
    <td>EventVAD <a href="http://arxiv.org/abs/2504.13092"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> </td>
    <td><img src="https://img.shields.io/badge/Explainable-green" height="20"></td>
    <td><img src="https://img.shields.io/badge/Train_Free-808080" height="20"></td>
    <td>feat</td>
    <td>ucfc</td>
    <td>xdv</td>
  </tr>
  <tr>
    <td>2025</td>
    <td>ProDisc-VAD <a href="https://github.com/modadundun/ProDisc-VAD"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a> <a href="http://arxiv.org/abs/2505.02179"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> <img src="https://img.shields.io/badge/uws4vad--blue?labelColor=purple" height="20"></td>
    <td>ac</td>
    <td><img src="https://img.shields.io/badge/Weak-FFA500" height="20"></td>
    <td>CLIP ViT-B/16</td>
    <td>ucfc</td>
    <td>xdv</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</tbody></table>


<details> 
  <summary> legend </summary> 

<!-- 
<a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch" height="20"></a>
<a href="#"><img src="https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red" height="20"></a> 
[![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](#)
[![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#)  
-->

- <img src="https://img.shields.io/badge/Weakly-FFA500" height="20">    <!-- Orange -->

- <img src="https://img.shields.io/badge/Open_World-00BFFF" height="20"> <!-- Deep Sky Blue -->

- <img src="https://img.shields.io/badge/Train_Free-808080" height="20"> <!-- Gray -->

- <img src="https://img.shields.io/badge/FineTune-rgb(21, 197, 197)" height="20">

- <img src="https://img.shields.io/badge/Explainable-green" height="20">

- <a href="#"><img src="https://img.shields.io/badge/uws4vad-purple" height="20"></a><img src="https://img.shields.io/badge/done-green" height="20"><img src="https://img.shields.io/badge/wip-yellow" height="20"><img src="https://img.shields.io/badge/roadmap-blue" height="20">
  - Badge is a link to file implementing the method (or better be config since it contains everything).

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



## Datasets
[//]: https://www.tablesgenerator.com/html_tables

<table><thead>
  <tr>
    <th rowspan="3">Year</th>
    <th rowspan="3">Dataset <br></th>
    <th colspan="2" rowspan="2">Video</th>
    <th rowspan="3"># Anomaly Types</th>
    <th colspan="4">GT</th>
  </tr>
  <tr>
    <th>Location</th>
    <th colspan="3">Text</th>
  </tr>
  <tr>
    <th>#</th>
    <th>Audio</th>
    <th>Level</th>
    <th>clip-event</th>
    <th>event-level</th>
    <th>video-level</th>
  </tr></thead>
<tbody>
  <tr>
    <td>2018</td>
    <td>UCF-Crime <a href="https://www.crcv.ucf.edu/projects/real-world/">:link:</a> <a href="https://github.com/WaqasSultani/AnomalyDetectionCVPR2018">OG:file_folder:</a> <a href="https://github.com/Roc-Ng/DeepMIL">Torch:file_folder:</a> <a href="https://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf">:newspaper:</a></td>
    <td></td>
    <td></td>
    <td>13</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2020</td>
    <td>XD-Violence <a href="https://roc-ng.github.io/XD-Violence/">:link:</a> <a href="https://github.com/Roc-Ng/XDVioDet">:file_folder:</a> <a href="https://arxiv.org/pdf/2007.04687">:newspaper:</a></td>
    <td></td>
    <td></td>
    <td>6</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td></td>
    <td>UCF_Crime Extra ++ <a href="https://github.com/hibrahimozturk/temporal_anomaly_detection">:file_folder:</a> <a href="https://arxiv.org/pdf/2104.06653">:newspaper:</a></td>
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
    <td>MSAD <a href="https://msad-dataset.github.io/">:link:</a> <a href="https://github.com/Tom-roujiang/MSAD">:file_folder:</a> <a href="https://arxiv.org/abs/2402.04857">:newspaper:</a></td>
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
    <td>CUVA/ECVA <a href="https://github.com/fesvhtr/CUVA">:file_folder:</a> <a href="https://github.com/Dulpy/ECVA">:file_folder:</a>  <a href="https://arxiv.org/pdf/2412.07183">:newspaper:</a></td>
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
    <td>HIVAU-70k <a href="https://www.tablesgenerator.com/html_tables">:link:</a> <a href="https://www.tablesgenerator.com/html_tables">:file_folder:</a> <a href="https://www.tablesgenerator.com/html_tables">:newspaper:</a></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  :
  <tr>
    <td></td>
    <td>TAU-106K <a href="https://www.tablesgenerator.com/html_tables">:link:</a> <a href="https://github.com/cool-xuan/TABot">:file_folder:</a> <a href="https://www.tablesgenerator.com/html_tables">:newspaper:</a></td>
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
    <td>___<a href="https://www.tablesgenerator.com/html_tables">:link:</a> <a href="https://www.tablesgenerator.com/html_tables">:file_folder:</a> <a href="https://www.tablesgenerator.com/html_tables">:newspaper:</a></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</tbody></table>

For a more detailed information/stats about datasets refer to [Awesome-Video-Anomaly-Detection](https://github.com/Junxi-Chen/Awesome-Video-Anomaly-Detection/tree/main) by [@Junxi-Chen](https://github.com/Junxi-Chen)

<!---
UCA [62] only provides clip-level captions, overlooking the understanding of anomalies across longer time spans. 
CUVA [9] and Hawk [45], on the other hand, only offer video-level instruction data, neglecting finer-grained visual perception and anomaly analysis. 
Our proposed HIVAU-70k takes a multi-temporal granularity perspective. It enables progressive and comprehensive learning, from short-term visual perception to long-term anomaly reasoning.

Methods,#Catogories,#Samples,Text(clip-level,event-level,video-level),TempAnno,MLLM-tuning  
UCA,13,23542,✓,✗,✗,✓,✗
LAVAD,N/A,N/A,✓,✗,✓,✗,✗
VAD-VideoLLama,13/7,2400,✗,✗,✓,✗,projection
CUVA,11,6000,✗,✗,✓,✗,✗ 
Hawk,-,16000,✗,✗,✓,✗,projection 
HIVAU-70k, 19,70000,✓,✓,✓,✓,LoRA


-->


## ECVA benchmark
<table>
    <thead>
        <tr>
            <th rowspan="2">Year</th>
            <th rowspan="2">Method<br></th>
            <th colspan="3">AnomEval</th>
        </tr>
        <tr>
            <th>Cause</th>
            <th>Description</th>
            <th>Effect</th>
    </tr>
    </thead>
    <tbody>
        <tr>
            <td>2023</td>
            <td>VILA [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)]() [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)]() </td>
            <td>0.35</td>
            <td>0.3749</td>
            <td>0.3188</td>
        </tr>
        <tr>
            <td>2024</td>
            <td>AnomShield</td>
            <td>0.33</td>
            <td>0.4057</td>
            <td>0.3509</td>
        </tr>
    </tbody>
</table>


## HIVAU-70K benchmark
<table>
    <thead>
        <tr>
            <th rowspan="2">Year</th>
            <th rowspan="2">Method<br></th>
            <th colspan="3">AnomEval</th>
        </tr>
        <tr>
            <th>Cause</th>
            <th>Description</th>
            <th>Effect</th>
    </tr>
    </thead>
    <tbody>
        <tr>
            <td>2023</td>
            <td>VILA [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)]() [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)]() </td>
            <td>0.35</td>
            <td>0.3749</td>
            <td>0.3188</td>
        </tr>
        <tr>
            <td>2024</td>
            <td>AnomShield</td>
            <td>0.33</td>
            <td>0.4057</td>
            <td>0.3509</td>
        </tr>
    </tbody>
</table>






### Other Collections/Resources

- Generalized Video Anomaly Event Detection: Systematic Taxonomy and Comparison of Deep Models [![](https://img.shields.io/badge/-9B59B6?logo=github&logoColor=white)](https://github.com/fudanyliu/GVAED) -> 

- Networking Systems for Video Anomaly Detection: A Tutorial and Survey [![](https://img.shields.io/badge/-9B59B6?logo=github&logoColor=white)](https://github.com/fdjingliu/NSVAD) -> A great starting point for VAD, covering supervision development and progression (U,Ws,Fu)

- [Quo Vadis, Anomaly Detection?  LLMs and VLMs in the Spotlight](http://arxiv.org/abs/2412.18298) [![](https://img.shields.io/badge/-black?logo=github&logoColor=white)](https://github.com/Darcyddx/VAD-LLM) -> A survey the integration of large language models (LLMs) and vision-language models (VLMs) in video anomaly detection (VAD)

- [Markdown-Cheatsheet](https://github.com/lifeparticle/Markdown-Cheatsheet)




<!---

https://www.tablesgenerator.com/html_tables


ICONS
  https://simpleicons.org/
  https://shields.io/docs/logos   https://badges.pages.dev/
  

  https://img.shields.io/badge/any_text-you_like-blue
  https://img.shields.io/badge/just%20the%20message-8A2BE2
-->



<!-- 
| Year | Method |  Feature & Supervision | UCF (AUCo,AUCa,FAR) | XDV (APo,APa,FAR) |
|------|--------|------------------------|---------------------|-------------------|
| 2018 | MIR  [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/Roc-Ng/DeepMIL) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) ![](https://img.shields.io/badge/uws4vad-purple) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) | 0.7541,-,- <br> 0.7792,-,- | -,-,- <br> -,-,- | 
| 2019 | GCN [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/jx-zhong-for-academic-purpose/GCN-Anomaly-Detection) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> TSN ![](https://img.shields.io/badge/W-FFA500) | 0.8192,-,- <br> 0.8212,-,- | -,-,- <br> -,-,- |
| 2019 | MA [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | PWC-OF ![](https://img.shields.io/badge/W-FFA500) | 0.7210,-,- | -,-,- |
| 2019 | TCN [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) | 0.7866,-,- | -,-,- |
| 2020 | SRG [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) | 0.7954,-,0.13 | -,-,- |
| 2020 | ARN [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.7571,-,- | -,-,- |
| 2020 | HLN/XDV [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/Roc-Ng/XDVioDet) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | 0.8244,-,- <br> -,-,- | -,-,- <br> 0.7864,-,- |
| 2020 | WSAL [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/ktr-hubrt/WSAL) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) <br> R(2+1)D ![](https://img.shields.io/badge/W-FFA500) <br> TSN ![](https://img.shields.io/badge/W-FFA500) | 0.7418,-,- <br> 0.7529,-,- <br> 0.8538,0.6738,- | -,-,- <br> -,-,- <br> -,-,- |
| 2020 | CLAWS [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) | 0.8303,-,0.12 | -,-,- |
| 2021 | MIST [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8140,-,2.19 <br> 0.8230,-,0.13 | -,-,- <br> -,-,- |
| 2021 | AVF [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | -,-,- | 0.8169,-,- |
| 2021 | RTFM [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/tianyu0207/RTFM) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) ![](https://img.shields.io/badge/uws4vad-purple) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8328,-,- <br> 0.8430,-,- | 0.7589,-,- <br> 0.7781,-,- |
| 2021 | XEL [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/sdjsngs/XEL-WSAD) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) | 0.8260,-,- | -,-,- |
| 2021 | CA [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/changsn/Contrastive-Attention-for-Video-Anomaly-Detection) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> TSN <br> I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8340,-,- <br> 0.8352,-,- <br> 0.8462,-,- | -,-,- <br> -,-,- <br> 0.7690,-,- |
| 2021 | MS-BS [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8353,-,- | -,-,- |
| 2021 | DAM [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=tensorflow)](https://github.com/snehashismajhi/DAM-Anomaly-Detection) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8267,-,0.3 | -,-,- |
| 2022 | CMALA [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/yujiangpu20/cma_xdVioDet) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) ![](https://img.shields.io/badge/uws4vad-purple) | I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | -,-,- | 0.8354,-,- |
| 2022 | CLAWS+ [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> 3DRN ![](https://img.shields.io/badge/W-FFA500) | 0.8337,-,0.11 <br> 0.8416,-,0.09 | -,-,- <br> -,-,- |
| 2022 | WSTR [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/justsmart/WSTD-VAD) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8317,-,- | -,-,- |
| 2022 | STA [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8160,-,- <br> 0.8300,-,- | -,-,- <br> -,-,- |
| 2022 | MSL [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/xidianai/MSL) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) <br> VSWIN | 0.8285,-,- <br> 0.8530,-,- <br> 0.8562,-,- | 0.7553,-,- <br> 0.7828,-,- <br> 0.7859,-,- |
| 2022 | SGMIR [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8170,-,- | -,-,- |
| 2022 | TCA [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8208,-,0.11 <br> 0.8375,-,0.05 | -,-,- <br> -,-,- |
| 2022 | MACIL [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/JustinYuu/MACIL_SD) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) ![](https://img.shields.io/badge/uws4vad-purple) | I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | -,-,- | 0.8340,-,- |
| 2022 | LAN [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8512,-,- | 0.8072,-,- |
| 2022 | ANM [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/sakurada-cnq/salient_feature_anomaly) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) ![](https://img.shields.io/badge/uws4vad-purple) | I3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | 0.8299,-,- <br> -,-,- | -,-,- <br> 0.8491,-,- |
| 2022 | MSAF [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/Video-AD/MSFA) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D+OF ![](https://img.shields.io/badge/W-FFA500) <br> I3D+OF+VGG ![](https://img.shields.io/badge/W-FFA500) | 0.8134,-,- <br> -,-,- | -,-,- <br> 0.8051,-,- |
| 2022 | TAI [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8573,-,- | -,-,- |
| 2022 | BSME [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8363,-,- | -,-,- |
| 2022 | MGFN [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/carolchenyx/MGFN) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) ![](https://img.shields.io/badge/uws4vad-purple) | I3D ![](https://img.shields.io/badge/W-FFA500) <br> VSwin ![](https://img.shields.io/badge/W-FFA500) | 0.8698,-,- <br> 0.8667,-,- | 0.7919,-,- <br> 0.8011,-,- |
| 2022 | CUN [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/ArielZc/CU-Net) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | 0.8622,-,- <br> -,-,- | 0.7874,-,- <br> 0.8143,-,- |
| 2022 | CLIP-TSA [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/joos2010kj/CLIP-TSA) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) ![](https://img.shields.io/badge/uws4vad-purple) | CLIP ![](https://img.shields.io/badge/W-FFA500) | 0.8758,-,- | 0.8219,-,- |
| 2023 | NGMIL [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8343,-,- <br> 0.8563,-,- | 0.7591,-,- <br> 0.7851,-,- |
| 2023 | URDMU [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/henrryzh1/UR-DMU) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | 0.8697,-,1.05 <br> -,-,- | 0.8166,-,0.65 <br> 0.8177,-,- |
| 2023 | UMIL [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/ktr-hubrt/UMIL) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](https://arxiv.org/pdf/2303.12369v1) | X-CLIP ![](https://img.shields.io/badge/W-FFA500) | 0.8675,0.6868,- | -,-,- |
| 2023 | LSTC [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/shengyangsun/LSTC_VAD) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8347,-,- <br> 0.8588,-,- | -,-,- <br> -,-,- |
| 2023 | BERTMIL [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/wjtan99/BERT_Anomaly_Video_Classification) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D+FLOW ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8671,-,- <br> -,-,- | -,-,- <br> 0.8210,-,- |
| 2023 | SLAMBS [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8619,-,- | 0.8423,-,- |
| 2023 | TEVAD [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/coranholmes/TEVAD) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | | | |
| 2023 | HYPERVD [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/xiaogangpeng/HyperVD) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | -,-,- | 0.8567,-,- |
| 2023 | PEL4VAD [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/yujiangpu20/PEL4VAD) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) | 0.8676,0.7224,0.43 | 0.8559,0.7026,0.57 |
| 2023 | VAR [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/Roc-Ng/VAR) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | | | |
| 2023 | CNN-VIT [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | C3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D ![](https://img.shields.io/badge/W-FFA500) <br> CLIP ![](https://img.shields.io/badge/W-FFA500) <br> C3D+CLIP ![](https://img.shields.io/badge/W-FFA500) <br> I3D+CLIP ![](https://img.shields.io/badge/W-FFA500) | 0.8578,-,- <br> 0.8650,-,- <br> 0.8763,-,- <br> 0.8802,-,- <br> 0.8897,-,- | -,-,- <br> -,-,- <br> -,-,- <br> -,-,- <br> -,-,- |
| 2023 | UCFA [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | | | |
| 2023 | SAA [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/2023-MindSpore-4/Code4/tree/main/WS-VAD-mindspore-main) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | 0.8619,0.6877,- <br> -,-,- | 0.8359,0.8419,- <br> 0.8423,-,- |
| 2023 | ANOMCLIP [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/lucazanella/AnomalyCLIP) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | ViT-B/16 ![](https://img.shields.io/badge/W-FFA500) | 0.8636,-,- | 0.7851,-,- |
| 2023 | MTDA [](#) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | -,-,- | 0.8444,-,- |
| 2023 | BNWVAD [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/cool-xuan/BN-WVAD) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) ![](https://img.shields.io/badge/uws4vad-purple) | I3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | 0.8724,0.7171,- <br> -,-,- | 0.8493,0.8545,- <br> 0.8526,-,- |
| 2023 | DEN [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/ArielZc/DE-Net) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D ![](https://img.shields.io/badge/W-FFA500) <br> I3D+VGG ![](https://img.shields.io/badge/W-FFA500) | 0.8633,-,- <br> -,-,- | 0.8166,-,- <br> 0.8313,-,- |
| 2023 | VADCLIP [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/nwpu-zxr/VadCLIP) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](#) | I3D+CLIP ![](https://img.shields.io/badge/W-FFA500) | 0.8802,-,- | 0.8451,-,- |
| 2024 | VAD-LLaAMa [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/ktr-hubrt/VAD-LLaMA) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](http://arxiv.org/abs/2401.05702) | | | |
| 2024 | GlanceVAD [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/pipixin321/GlanceVAD) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](http://arxiv.org/abs/2403.06154) ![](https://img.shields.io/badge/uws4vad-purple) | | | |
| 2024 | LAVAD [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/lucazanella/lavad) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](http://arxiv.org/abs/2404.01014) | | | |
| 2024 | MSBT [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/shengyangsun/MSBT) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](http://arxiv.org/abs/2405.05130) | I3D+VGG ![](https://img.shields.io/badge/W-FFA500) <br> I3D+VGG+TV-L1 ![](https://img.shields.io/badge/W-FFA500)| -,-,- <br> -,-,- | 0.8254,-,- <br> 0.8432,-,- |
| 2024 | HAWK [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/jqtangust/hawk) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](http://arxiv.org/abs/2405.16886) | | | |
| 2024 | PEMIL [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/Junxi-Chen/PE-MIL) [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](https://ieeexplore.ieee.org/document/10657732/) | | | |
| 2024 | Holmes-VAD [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/pipixin321/HolmesVAD)[![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](http://arxiv.org/abs/2406.12235) | | | |
| 2024 | Holmes-VAU [![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=pytorch)](https://github.com/pipixin321/HolmesVAU)[![](https://img.shields.io/badge/-rgba(0,0,0,0)?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2412.06171) | | | |
||||||
-->