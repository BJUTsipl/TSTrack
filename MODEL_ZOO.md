# Model Zoo

Here we provide the performance of the SUTrack on multiple tracking benchmarks and the corresponding raw results. 
The model weights are also given by the links.


## SUTrack Models

<table>
  <tr>
    <th>Model</th>
    <th>LaSOT<br>AUC (%)</th>
    <th>LaSOText<br>AUC (%)</th>
    <th>TrackingNet<br>AUC (%)</th>
    <th>GOT-10k<br>AO (%)</th>
    <th>VOT-RGBD22<br>EAO (%)</th>
    <th>DepthTrack<br>F-score (%)</th>
    <th>LasHeR<br>AUC (%)</th>
    <th>RGBT234<br>MSR (%)</th>
    <th>VisEvent<br>AUC</th>
    <th>TNL2K<br>AUC</th>
    <th>OTB99<br>AUC (%)</th>
    <th>Models</th>
  </tr>
  <tr>
    <td>SUTrack-T224</td>
    <td>69.6</td>
    <td>50.2</td>
    <td>82.7</td>
    <td>72.7</td>
    <td>68.1</td>
    <td>61.7</td>
    <td>53.9</td>
    <td>63.8</td>
    <td>58.8</td>
    <td>60.9</td>
    <td>67.4</td>
    <td><a href="https://pan.baidu.com/s/1XizoKl6zduj6l8Tb3SQi2g?pwd=jqpb">[Download]</a></td>
  </tr>
  <tr>
    <td>SUTrack-B224</td>
    <td>73.2</td>
    <td>53.1</td>
    <td>85.7</td>
    <td>77.9</td>
    <td>76.5</td>
    <td>65.1</td>
    <td>59.9</td>
    <td>69.5</td>
    <td>62.7</td>
    <td>65.0</td>
    <td>70.8</td>
    <td><a href="https://pan.baidu.com/s/1XizoKl6zduj6l8Tb3SQi2g?pwd=jqpb">[Download]</a></td>
  </tr>
  <tr>
    <td>SUTrack-B384</td>
    <td>74.4</td>
    <td>52.9</td>
    <td>86.5</td>
    <td>79.3</td>
    <td>76.6</td>
    <td>64.4</td>
    <td>60.9</td>
    <td>59.2</td>
    <td>63.4</td>
    <td>65.6</td>
    <td>69.7</td>
    <td><a href="https://pan.baidu.com/s/1XizoKl6zduj6l8Tb3SQi2g?pwd=jqpb">[Download]</a></td>
  </tr>
  <tr>
    <td>SUTrack-L224</td>
    <td>73.5</td>
    <td>54.0</td>
    <td>86.5</td>
    <td>81.0</td>
    <td>76.4</td>
    <td>64.3</td>
    <td>61.9</td>
    <td>70.8</td>
    <td>64.0</td>
    <td>66.7</td>
    <td>72.7</td>
    <td><a href="https://pan.baidu.com/s/1XizoKl6zduj6l8Tb3SQi2g?pwd=jqpb">[Download]</a></td>
  </tr>
  <tr>
    <td>SUTrack-L384</td>
    <td>75.2</td>
    <td>53.6</td>
    <td>87.7</td>
    <td>81.5</td>
    <td>76.6</td>
    <td>66.4</td>
    <td>61.9</td>
    <td>70.3</td>
    <td>63.8</td>
    <td>67.9</td>
    <td>71.2</td>
    <td><a href="https://pan.baidu.com/s/1XizoKl6zduj6l8Tb3SQi2g?pwd=jqpb">[Download]</a></td>
  </tr>
</table>

[Download Models](https://pan.baidu.com/s/1XizoKl6zduj6l8Tb3SQi2g?pwd=jqpb)

The downloaded checkpoints should be organized in the following structure
   ```
   ${SUTrack_ROOT}
    -- checkpoints
        -- train
            -- sutrack
                -- sutrack_b256
                    SUTRACK_ep0180.pth.tar
                -- sutrack_b384
                    SUTRACK_ep0180.pth.tar
                -- sutrack_l256
                    SUTRACK_ep0180.pth.tar
                -- sutrack_l384
                    SUTRACK_ep0180.pth.tar
   ```

## Raw Results
Raw results are being prepared.
Running the model directly should yield the same results as the paper.
