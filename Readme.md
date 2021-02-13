# TEST 1
In this test I have the following idea:
* Instead of trying to detect the keypoints find the ```arms```, ```legs```, ```heads``` and ```bodies``` seperately (maybe use a heatmap).
  * You may want to use transfer learning, utilize the ResNet50 or VGG19 for detecting the body parts of the people.

* Then from the parts try to merge them. Find the belongings.
  * This approach has this advandage. If a part could not be seen we can deduce to, that part may be behind of something. So thet we can compute relative depth and aid our 3D approximation.