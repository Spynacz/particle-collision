#include <vector>

using namespace std;

// Original author: Andrew Noske
// https://andrewnoske.com/wiki/Code_-_heatmaps_and_color_gradients
class ColorGradient {
   private:
    struct ColorPoint  // Internal class used to store colors at different
                       // points in the gradient.
    {
        float r, g, b;  // Red, green and blue values of our color.
        float
            val;  // Position of our color along the gradient (between 0 and 1).
        ColorPoint(float red, float green, float blue, float value)
            : r(red), g(green), b(blue), val(value) {}
    };
    vector<ColorPoint> color;  // An array of color points in ascending value.

   public:
    //-- Default constructor:
    ColorGradient() { createDefaultHeatMapGradient(); }

    //-- Inserts a new color point into its correct position:
    void addColorPoint(float red, float green, float blue, float value) {
        for (unsigned int i = 0; i < color.size(); i++) {
            if (value < color[i].val) {
                color.insert(color.begin() + i,
                             ColorPoint(red, green, blue, value));
                return;
            }
        }
        color.push_back(ColorPoint(red, green, blue, value));
    }

    //-- Inserts a new color point into its correct position:
    void clearGradient() { color.clear(); }

    //-- Places a 5 color heapmap gradient into the "color" vector:
    void createDefaultHeatMapGradient() {
        color.clear();
        color.push_back(ColorPoint(0, 0, 1, 0.0f));   // Blue.
        color.push_back(ColorPoint(0, 1, 1, 0.25f));  // Cyan.
        color.push_back(ColorPoint(0, 1, 0, 0.5f));   // Green.
        color.push_back(ColorPoint(1, 1, 0, 0.75f));  // Yellow.
        color.push_back(ColorPoint(1, 0, 0, 1.0f));   // Red.
    }

    void mintHeatMapGradient() {
        const unsigned int numberOfColors = 7;
        float step = 1.0f / (numberOfColors - 1);
        color.clear();
        color.push_back(ColorPoint(228, 241, 225, step * 0));
        color.push_back(ColorPoint(180, 217, 204, step * 1));
        color.push_back(ColorPoint(137, 192, 182, step * 2));
        color.push_back(ColorPoint(99, 166, 160, step * 3));
        color.push_back(ColorPoint(68, 140, 138, step * 4));
        color.push_back(ColorPoint(40, 114, 116, step * 5));
        color.push_back(ColorPoint(13, 88, 95, step * 6));

        rescaleColorPoints();
    }

    void viridisHeatMap() {
        const unsigned int numberOfColors = 8;
        float step = 1.0f / (numberOfColors - 1);

        color.clear();
        color.push_back(ColorPoint(0.267004, 0.004874, 0.329415, step * 0));
        color.push_back(ColorPoint(0.275191, 0.194905, 0.496005, step * 1));
        color.push_back(ColorPoint(0.212395, 0.359683, 0.55171, step * 2));
        color.push_back(ColorPoint(0.153364, 0.497, 0.557724, step * 3));
        color.push_back(ColorPoint(0.122312, 0.633153, 0.530398, step * 4));
        color.push_back(ColorPoint(0.288921, 0.758394, 0.428426, step * 5));
        color.push_back(ColorPoint(0.626579, 0.854645, 0.223353, step * 6));
        color.push_back(ColorPoint(0.993248, 0.906157, 0.143936, step * 7));
    }

    void tealRoseHeatMap() {
        const unsigned int numberOfColors = 7;
        float step = 1.0f / (numberOfColors - 1);

        color.clear();
        color.push_back(
            ColorPoint(0.0, 0.5764705882352941, 0.5725490196078431, step * 0));
        color.push_back(ColorPoint(0.4470588235294118, 0.6666666666666666,
                                   0.6313725490196078, step * 1));
        color.push_back(ColorPoint(0.6941176470588235, 0.7803921568627451,
                                   0.7019607843137254, step * 2));
        color.push_back(ColorPoint(0.9450980392156862, 0.9176470588235294,
                                   0.7843137254901961, step * 3));
        color.push_back(ColorPoint(0.8980392156862745, 0.7254901960784313,
                                   0.6784313725490196, step * 4));
        color.push_back(ColorPoint(0.8509803921568627, 0.5372549019607843,
                                   0.5803921568627451, step * 5));
        color.push_back(ColorPoint(0.8156862745098039, 0.34509803921568627,
                                   0.49411764705882355, step * 6));
    }

    // If you declare Heat Map ass rgb in range 0-255 you need to rescale it to
    // 0-1
    void rescaleColorPoints() {
        for (auto& colorPoint : color) {
            colorPoint.r /= 255;
            colorPoint.g /= 255;
            colorPoint.b /= 255;
        }
    }

    //-- Inputs a (value) between 0 and 1 and outputs the (red), (green) and
    //(blue)
    //-- values representing that position in the gradient.
    void getColorAtValue(const float value, float& red, float& green,
                         float& blue) {
        if (color.size() == 0) return;

        for (unsigned int i = 0; i < color.size(); i++) {
            ColorPoint& currC = color[i];
            if (value < currC.val) {
                ColorPoint& prevC = color[max(0, (int)i - 1)];
                float valueDiff = (prevC.val - currC.val);
                float fractBetween =
                    (valueDiff == 0) ? 0 : (value - currC.val) / valueDiff;
                red = (prevC.r - currC.r) * fractBetween + currC.r;
                green = (prevC.g - currC.g) * fractBetween + currC.g;
                blue = (prevC.b - currC.b) * fractBetween + currC.b;
                return;
            }
        }
        red = color.back().r;
        green = color.back().g;
        blue = color.back().b;
        return;
    }
};