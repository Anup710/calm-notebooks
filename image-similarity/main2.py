import cv2

orb = cv2.ORB_create()

# p1 = 'images/kiernan_shipka.jpg'
# p2 = 'images/mckenna_grace.jpg'

# Load images in grayscale
img1 = cv2.imread("images/kiernan_shipka.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/mckenna_grace.png", cv2.IMREAD_GRAYSCALE)

# img1 = cv2.imread("images/beckham.png", cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread("images/denzel_washington.png", cv2.IMREAD_GRAYSCALE)


# Resize images to the same shape (if needed)
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Detect and compute keypoints & descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)
similarity_score = len(matches)
print(f"Feature Matching Similarity Score: {similarity_score}")