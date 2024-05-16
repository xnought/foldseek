# Code Review / Inspection

This report was created after reviewing the [`lib/3di/`](../lib/3di/) files.

**Overview**: The code in the folder converts a three-dimensional protein structure into a string that encodes local 3D structure. Thus the name 3Di. It uses a special representation with angles and shortest distances and a ML models to create the string.

An example protein input is shown in [`d1qd6.1.ent`](../lib/3di/example/d1qd6.1.ent) where the 3D coordinates are shown. This is what is converted into a string like AAPFLRG... as the 3Di sequence.

**Review criteria**: In this review I will address each file and find issues or improvements. I will make use of the taxonomy of bad code smells https://mmantyla.github.io/BadCodeSmellsTaxonomy for a common language.

I will also look for consistent code styling and where the code can be more easily read. Since a paper accompanies this code, I will see if the notation matches too https://www.nature.com/articles/s41587-023-01770 when there is math.

## Overall

I wrote this section after I did the reviews in the next sections. I think the code is overall very well written. However, many of the choices seem arbitrary and very hard to read when the math gets dense.

I recommend the devs add consistent file imports, semantically meaningful variable names, comments that describe the math notation, and function comment headers which describe what the function does. Otherwise, other developers will have a very hard time reading and interacting with the code.

My detailed review is below:

## [`structureto3di.cpp`](../lib/3di/structureto3di.cpp)

### [Line 1](../lib/3di/structureto3di.cpp#L1)

```cpp
#include <string.h> // Line 1

#include <iostream>
#include <vector>
#include <cmath>
#include "structureto3di.h"
#include "encoder_weights_3di.kerasify.h"
```

The import styling is inconsistent with the other files. There is a space and the files are not sorted with any meaning that would make this file easier to read.

One suggestion is to first show all imported C++ libraries, then C (i.e. `string.h`), then the custom header files. 


### [Line 23](../lib/3di/structureto3di.cpp#L23)

The naming could be more specific and consistent with how they describe their process. 
For example

```cpp
Vec3 StructureTo3DiBase::norm(Vec3 a){ // Line 23
    double len = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    a.x = a.x / len;
    a.y = a.y / len;
    a.z = a.z / len;
    return a;
}
```

describes the Euclidean or 2-norm. There are many other types of norms, so I think they should be more specific, not less specific and instead name the function `norm2` or something like that. Also the function does much more than just compute the norm, it also divides by the norm which is not how people usually call it.

Additionally, adding the math notation they used in their paper would be helpful. For example they could add a comment showing `// a / ||a||_2 = a / \sqrt{a_x^2 + a_y^2 + a_z^2}` (which renders to $a / ||a||_2 = a / \sqrt{a_x^2 + a_y^2 + a_z^2}$) within the function.

I should also note that `len` can be 0 when x, y, and z are all 0. That would result in division by 0 error.

### [Line 61](../lib/3di/structureto3di.cpp#L61)

```cpp
Vec3 v4 = add( // Line 61
		scale(v1, -1/3.0), scale(sub(scale(u1, -1/2.0),
		scale(u2, sqrt(3)/2.0)), sqrt(8)/3.0)
		);
```
There are too many operations for this to be readable. Either add math notation in a comment that is more meaningful or break up the operations. I think the constants could also be named so that we know what's going on.

Namely, the square roots and divisions are confusing to the reader. I believe that even the person who wrote this code wouldn't know what's going on if they came back later.


### [Line 73](../lib/3di/structureto3di.cpp#L73)

```cpp
Vec3 StructureTo3DiBase::calcVirtualCenter(Vec3 ca, Vec3 cb, Vec3 n, // Line 73
                                       double alpha, double beta, double d){
// ... rest not shown
```
The angle values have math names, which is not too bad (alpha and beta), but there is no indication if the variables are in degrees or radians. In fact, this function ends up converting alpha and beta to radians. So this function needs better input variable naming or better comments.

### [Line 89](../lib/3di/structureto3di.cpp#L89)

```cpp
v = add(add(scale(v, cos(alpha)), scale(cross(k, v), sin(alpha))), // Line 89
        scale(scale(k, dot(k, v)), 1 - cos(alpha)));
```

The same goes for line 89 where the operations are so nested that no person could understand what is going on unless they wrote the code themselves. To add more readability, add math notation or break up the operation.

### [Line 101](../lib/3di/structureto3di.cpp#L101)

```cpp
double StructureTo3DiBase::calcDistanceBetween(Vec3 & a, Vec3 & b){ // Line 101
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return sqrt(dx*dx + dy*dy + dz*dz);
}
```

To match their research paper, I would rename this function to be more specific, like Euclidean distance or distance computed with 2-norm.


### [Line 136](../lib/3di/structureto3di.cpp#L136)

This function might return -1 and distance if INFINITY, but you wouldn't know that if you were just using the function. I would put in a comment that this function may not return what you expect if nothing is found.

### [Line 189](../lib/3di/structureto3di.cpp#L189)

```cpp
StructureTo3Di::Feature StructureTo3Di::calcFeatures(Vec3 * ca, int i, int j){ // Line 189
    Vec3 u1 = norm(sub(ca[i],       ca[i - 1]));
    Vec3 u2 = norm(sub(ca[i + 1],   ca[i]));
    Vec3 u3 = norm(sub(ca[j],       ca[j - 1]));
    Vec3 u4 = norm(sub(ca[j + 1],   ca[j]));
    Vec3 u5 = norm(sub(ca[j],       ca[i]));

    double features[Alphabet3Di::FEATURE_CNT];
    features[0] = dot(u1, u2);
    features[1] = dot(u3, u4);
    features[2] = dot(u1, u5);
    features[3] = dot(u3, u5);
    features[4] = dot(u1, u4);
    features[5] = dot(u2, u3);
    features[6] = dot(u1, u3);
    features[7] = calcDistanceBetween(ca[i], ca[j]);
    features[8] = copysign(fmin(fabs(j - i), 4), j - i); // clip j-i to [-4, 4]
    features[9] = copysign(log(fabs(j - i) + 1), j - i );
    return Feature(features);
}
```

This is probably the most important function of the entire file, but it is utterly confusing to read. First, the features should be labeled for which features corresponds to what instead of just indexing into a features array.

Additionally, I have no clue what each operation is doing if I were reading this for the first time. I think they should use the same notation they used in their paper which could tell the developer that the dot products are for angle computation and the other features are distances.

### [Line 275](../lib/3di/structureto3di.cpp#L275)

Overall this function is very well written with functions describing the high-level operations. However there are a few operations that without comments or better naming, I have no clue what is going on.

## [`structureto3di.h`](../lib/3di/structureto3di.h)

### [Line 8](../lib/3di/structureto3di.h#L8)

```cpp
namespace Alphabet3Di{
    static const size_t CENTROID_CNT = 20;
    static const char INVALID_STATE = 2; // assign invalid residues to coil state
    const double DISTANCE_ALPHA_BETA = 1.5336;
    const double PI = 3.14159265359;
    static const size_t FEATURE_CNT = 10;
    static const size_t EMBEDDING_DIM = 2;
    static const struct {
        double alpha, beta, d;
    } VIRTUAL_CENTER = { 270, 0, 2 };
//... rest not shown
```

Some of the constants are self-explanatory, but others are unclear what they are. Like what the features are that result in 10 of them, and what embedding dim is. More comments would greatly improve the readability or better variable naming.

## [`structureto3diseqdist.cpp`](../lib/3di/structureto3diseqdist.cpp)

### [Line 1](../lib/3di/structureto3diseqdist.cpp#L1)

```cpp
#include <string.h> // Line 1

#include <iostream>
#include <vector>
#include <math.h>
#include <limits.h>
#include "structureto3diseqdist.h"
```

Like I said before, consistent importing would make things easier to read.


### [Line 35](../lib/3di/structureto3diseqdist.cpp#L35)

```cpp
char * StructureTo3diSeqDist::structure2states(Vec3 * ca, Vec3 * n, // Line 35
                                        Vec3 * c, Vec3 * cb,
                                        size_t len){
// ... rest not shown
```

Again, the naming convetion for variables makes sense for math notation, but is very hard to read. What is ca? n? c? cb? The notation is described in the paper, but if there isn't a comment, no one would know (which there isn't).

## [`structureto3diseqdist.h`](../lib/3di/structureto3diseqdist.h)

### [Line 7](../lib/3di/structureto3diseqdist.h#L7)

```cpp
namespace Alphabet3diSeqDist{ // Line 7
    static const size_t CENTROID_CNT = 20;
    static const char INVALID_STATE = CENTROID_CNT;
    const int centroids[CENTROID_CNT] =
        {-284,-147,-83,-52,-33,-21,-13,-7,-4,-3,-1,1,3,7,13,24,40,68,123,250};
}
```

Why invalid state needs to have a centroid count and why the centroid counts are in this sequence of numbers is a mystery. Too many arbitrary choices, need to name things or comment better. 