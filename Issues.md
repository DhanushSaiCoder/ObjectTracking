# Issues

1. **[TEST: drone.mp4]** ISSUE: Tracking a dot area and verifyying correctly and going to 'low similarity' mode, but not actually searching for the real target object. 

   > Fix locked tracking of a false positive(mostly a small dot region tracking) -- improve verification process
    ---

2. **[TEST: person_sim.mp4]** ISSUE: More false positive score most likely due to less similarity matching score threshold. 

   > Increase similarity threshold and test, even with other footages to observe the effectors of new tuned parameter
   ---

3. **[TEST: drone.mp4]** ISSUE: Searching is happening only on very less area which is leading to failure if the object reappears far away from where it has disappeared

   > Increase the area of searching while tracker is in SEARCHING mode, but the thing to look carefully into is that the tracker searching in larger area might track a very similar object
   ---
